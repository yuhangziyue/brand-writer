"""HTTP 接口：SSE 流、会话查询、页面、回放模式（含 replay_unmatched）、healthz、静态前端挂载。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ragwriter.agent.intent import SUGGESTED_TOPICS
from ragwriter.agent.replay import FixtureLibrary, export_run, similarity
from ragwriter.agent.testing import make_fake_deps
from ragwriter.api.app import create_app

BLOG = "写一篇官网 Blog：为什么中国出海品牌需要进行 GEO 优化？"


def parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture
def paths(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>ragwriter</title>", encoding="utf-8")
    return {"sessions_db": tmp_path / "s.db", "fixtures_dir": tmp_path / "fx", "web_dir": web,
            "pages_jsonl": tmp_path / "none.jsonl"}


@pytest.fixture
def live(paths):
    return TestClient(create_app(deps=make_fake_deps(), **paths))


def test_chat_sse_stream(live):
    # 10-23：写作前先审核主题、与用户确认——第一轮只出确认卡，不检索不写作
    r = live.post("/v1/chat", json={"message": BLOG})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    evs = parse_sse(r.text)
    assert evs[0][0] == "session" and evs[1][0] == "intent" and evs[-1][0] == "done"
    types = {t for t, _ in evs}
    assert "confirm" in types and "retrieval" not in types and evs[-1][1]["awaiting_confirmation"] is True
    sid = evs[0][1]["session_id"]

    # 确认卡的按钮：直接按动作处理，进入写作
    evs1 = parse_sse(live.post("/v1/chat", json={"session_id": sid, "message": "确认",
                                                 "options": {"confirm": "accept"}}).text)
    assert {"plan", "retrieval", "delta", "verify", "section_final"} <= {t for t, _ in evs1}

    # 再出一版 FAQ：继承主题（这里跳过确认，只测继承）
    evs2 = parse_sse(live.post("/v1/chat", json={"session_id": sid, "message": "再出一版 FAQ",
                                                 "options": {"confirmed": True}}).text)
    assert evs2[1][1]["inherited"]["topic"]
    s = live.get(f"/v1/sessions/{sid}").json()
    assert [t["role"] for t in s["turns"]] == ["user", "assistant"] * 3
    assert len(s["artifacts"]) == 2 and s["slots"]["genre"] == "faq"
    assert live.get("/v1/sessions/nope").status_code == 404

    run = live.get(f"/v1/runs/{evs[-1][1]['run_id']}").json()
    assert run["meta"]["message"] == BLOG and run["events"][-1]["type"] == "done"


def test_generate_wrapper(live):
    evs = parse_sse(live.post("/v1/generate", json={"topic": "GEO 合规", "genre": "faq"}).text)
    assert evs[1][1]["source"] == "forced" and evs[1][1]["genre"] == "faq"
    assert evs[-1][0] == "done"


def test_pages_and_helpers(live):
    p = live.get("/v1/pages/10").json()
    assert p["page_no"] == 10 and p["image_url"] == "/v1/pages/10/image" and "image_path" not in p
    assert {f["fact_id"] for f in p["facts"]} == {"p10-f01", "p10-f02"}
    assert live.get("/v1/pages/99").status_code == 404
    assert live.get("/v1/suggestions").json() == {"suggestions": list(SUGGESTED_TOPICS)}
    h = live.get("/healthz").json()
    assert h["ok"] is True and h["mode"] == "live" and h["store"] is True
    assert live.get("/v1/mode").json()["mode"] == "live"


def test_static_web_mounted_after_api(live):
    assert "ragwriter" in live.get("/").text
    assert live.get("/healthz").json()["ok"] is True  # API 路由优先于静态挂载


def test_replay_mode_without_key(paths):
    # 先用 live 跑一次并导出为 fixture
    live_app = create_app(deps=make_fake_deps(), **paths)
    live = TestClient(live_app)
    evs = parse_sse(live.post("/v1/chat", json={"message": BLOG, "options": {"confirmed": True}}).text)
    export_run(paths["sessions_db"], evs[-1][1]["run_id"], "blog_geo", paths["fixtures_dir"])

    # 无 key：client=None ⇒ 回放
    replay = TestClient(create_app(deps=make_fake_deps(client=None), **paths))
    assert replay.get("/healthz").json()["mode"] == "replay"
    mode = replay.get("/v1/mode").json()
    assert mode["mode"] == "replay" and mode["recorded_topics"] == [BLOG] and mode["model"] == "fake-model"

    r = replay.post("/v1/chat", json={"message": BLOG + " ", "options": {"replay_speed": 0}})
    got = parse_sse(r.text)
    assert [t for t, _ in got] == [t for t, _ in evs]
    assert got[-1][1]["replay"] is True and got[-1][1]["fixture"] == "blog_geo"
    assert got[-1][1]["final_text"] == evs[-1][1]["final_text"]

    # /v1/generate 也能按主题命中录制
    got_g = parse_sse(replay.post("/v1/generate", json={"topic": "为什么中国出海品牌需要进行 GEO 优化？",
                                                        "options": {"replay_speed": 0}}).text)
    assert got_g[-1][1]["replay"] is True and not got_g[-1][1].get("error")

    # 没录过的主题：明确报 replay_unmatched，不拿别的 fixture 冒充
    miss = parse_sse(replay.post("/v1/chat", json={"message": "写一篇 FAQ：GEO 监控系统",
                                                   "options": {"replay_speed": 0}}).text)
    err = next(d for t, d in miss if t == "error")
    assert err["reason"] == "replay_unmatched" and err["recorded_topics"] == [BLOG]
    assert miss[-1][0] == "done" and miss[-1][1]["replay"] is True and miss[-1][1]["final_text"] == ""


def test_replay_no_fixtures(paths):
    replay = TestClient(create_app(deps=make_fake_deps(client=None), **paths))
    evs = parse_sse(replay.post("/v1/chat", json={"message": BLOG, "options": {"replay_speed": 0}}).text)
    assert any(t == "error" and d["reason"] == "replay_unmatched" for t, d in evs)


def test_replay_timing_respects_speed(tmp_path):
    fx = tmp_path / "fx"
    fx.mkdir()
    lines = [{"_meta": {"message": "第10页讲了什么？", "intent": "qa"}},
             {"t": 0, "type": "session", "data": {"session_id": "s1"}},
             {"t": 400, "type": "delta", "data": {"text": "a"}},
             {"t": 9000, "type": "done", "data": {}}]
    (fx / "qa.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8")
    from ragwriter.agent.replay import replay_stream

    sleeps: list[float] = []
    evs = list(replay_stream(FixtureLibrary(fx), "第10页讲了什么", speed=2, session_id="new", sleep=sleeps.append))
    assert sleeps == [0.2, 0.75]  # 400ms/2；长停顿截到 1500ms 再 /2
    assert evs[0].data["session_id"] == "new" and evs[-1].data["replay"] is True


def test_similarity_is_strict():
    assert similarity(BLOG, BLOG.replace("？", "")) == 1.0
    assert similarity(BLOG, "写一篇 FAQ：GEO 监控系统") < 0.5
