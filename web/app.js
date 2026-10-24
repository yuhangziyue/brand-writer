/* 品宣写作台 · Demo 前端（原生 JS，无构建、无外部依赖）
 *
 * 结构：
 *   1. 配置与常量
 *   2. 工具函数
 *   3. 后端访问（含 ?mock=1 的假数据通道）
 *   4. SSE 解析
 *   5. handleEvent —— 所有事件字段的解析都集中在这里，接口对齐只改这一处
 *   6. 渲染：意图标签 / 大纲 / 小节与句子三态 / 引用角标 / 资料边界 / 校验摘要 / 拒答
 *   7. 页面面板（点角标看原页 + 高亮框）
 *   8. 写作台、追问、回放模式、注入开关
 */
'use strict';

/* ───────────── 1. 配置与常量 ───────────── */
const QS = new URLSearchParams(location.search);
const CFG = {
  mock: QS.get('mock') === '1',
  mockReplay: QS.get('replay') === '1',            // ?mock=1&replay=1 预览回放横幅与下拉
  mockIntervalMs: 50 / Math.max(0.25, Number(QS.get('speed') || 1)),
  holdMs: 1500,                                    // 被拦截句红色删除线保留时长（M3）
  flashMs: 3000,                                   // 改写句黄色底高亮时长
  pageW: 960, pageH: 540,                          // 文本块坐标系（PDF 点）
  forceIntent: false,                              // true 时写作台走 options.force_intent，跳过意图识别
};
const GENRE_LABEL = { blog: 'Blog', faq: 'FAQ', brand_intro: '品牌介绍', product_intro: '产品介绍' };
const GENRE_VERB = { blog: '写一篇官网 Blog', faq: '写一组官网 FAQ', brand_intro: '写一篇品牌介绍', product_intro: '写一篇产品介绍' };
const FALLBACK_SUGGESTIONS = [
  '什么是 LLM-原生 GEO，它和传统 GEO 有什么不同？',
  'AI 搜索成为新流量入口，品牌为什么要被 AI「记住」？',
  'GEO 为什么必须走合规路线？',
];
const CHECK_LABEL = { V1: '引用编号有效', V2: '数字与专名出现在原文', V3: '数字挂对对象', V4: '事实句带出处', V5: '片段支持本句' };
const DELETED_PLACEHOLDER = '（此句未通过校验，已删除，详见校验报告）';
const CITE_RE = /\[([^\[\]]*?S\d+[^\[\]]*?)\]/g;

const app = {
  mode: null,          // GET /v1/mode 的结果
  sessionId: null,
  busy: false,
  abort: null,
  runId: null,
  turns: new Map(),
  pageCache: new Map(),
  seq: 0,
};

/* ───────────── 2. 工具函数 ───────────── */
const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}
/** 校验原因转人话：去掉「V3: 」这类编号前缀（编号只进校验报告）。 */
function humanReason(reason) {
  return String(reason || '').replace(/\bV[1-5]\s*[:：]\s*/g, '').trim() || '未通过校验';
}
function sidsIn(text) {
  const out = [];
  String(text || '').replace(CITE_RE, (_, grp) => { (grp.match(/S\d+/g) || []).forEach((s) => { if (!out.includes(s)) out.push(s); }); return ''; });
  return out;
}
function stripCites(text) { return String(text || '').replace(CITE_RE, '').trim(); }
function fmtMs(ms) { return ms == null ? '—' : (ms >= 1000 ? (ms / 1000).toFixed(1) + ' 秒' : ms + ' 毫秒'); }
function fmtNum(n) { return n == null ? '—' : Number(n).toLocaleString('zh-CN'); }
function topicText(item) { return typeof item === 'string' ? item : (item && (item.label || item.message || item.topic || item.title)) || ''; }

/* ───────────── 3. 后端访问 ───────────── */
async function getJSON(url) {
  const r = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  return r.json();
}
async function fetchMode() {
  if (CFG.mock) {
    return CFG.mockReplay
      ? { mode: 'replay', model: 'deepseek-v4-pro（mock）', run_id: 'r_mock01', mock: true,
          recorded_topics: ['写一篇官网 Blog：为什么中国出海品牌需要进行 GEO 优化？', '再出一版 FAQ', '万悉科技的融资情况'] }
      : { mode: 'live', model: 'mock 数据', mock: true, recorded_topics: [] };
  }
  return getJSON('/v1/mode');
}
async function fetchSuggestions() {
  if (CFG.mock) return FALLBACK_SUGGESTIONS;
  const d = await getJSON('/v1/suggestions');
  const list = Array.isArray(d) ? d : (d && (d.suggestions || d.topics)) || [];
  return list.length ? list : FALLBACK_SUGGESTIONS;
}
async function fetchPage(n) {
  if (app.pageCache.has(n)) return app.pageCache.get(n);
  const url = CFG.mock ? `mock/pages/p${String(n).padStart(2, '0')}.json` : `/v1/pages/${n}`;
  const p = getJSON(url);
  app.pageCache.set(n, p);
  p.catch(() => app.pageCache.delete(n));
  return p;
}
function mockFileFor(message) {
  if (/融资/.test(message)) return 'mock/refuse.sse';
  if (/FAQ|faq|常见问题/.test(message)) return 'mock/faq.sse';
  return 'mock/demo.sse';
}

/**
 * 发起一轮对话：POST /v1/chat → text/event-stream。
 * onEvent(type, data) 按到达顺序回调；返回时流已结束。
 */
async function streamChat(body, onEvent, signal) {
  const parser = createSSEParser(onEvent);
  if (CFG.mock) {
    const r = await fetch(mockFileFor(body.message), { signal });
    const text = await r.text();
    const events = [];
    const p2 = createSSEParser((t, d) => events.push([t, d]));
    p2.push(text); p2.end();
    for (const [t, d] of events) {
      if (signal && signal.aborted) return;
      await sleep(CFG.mockIntervalMs);
      onEvent(t, d);
    }
    return;
  }
  const r = await fetch('/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) {
    let msg = `HTTP ${r.status}`;
    try { const j = await r.json(); msg = j.detail || j.reason || msg; } catch (_) { /* 非 JSON */ }
    onEvent('error', { stage: 'http', reason: msg, fatal: true });
    return;
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder('utf-8');
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    parser.push(dec.decode(value, { stream: true }));
  }
  parser.push(dec.decode());
  parser.end();
}

/* ───────────── 4. SSE 解析 ───────────── */
function createSSEParser(onEvent) {
  let buf = '';
  function parseBlock(block) {
    let type = 'message';
    const data = [];
    for (const line of block.split('\n')) {
      if (!line || line.startsWith(':')) continue;
      const i = line.indexOf(':');
      const field = i < 0 ? line : line.slice(0, i);
      let val = i < 0 ? '' : line.slice(i + 1);
      if (val.startsWith(' ')) val = val.slice(1);
      if (field === 'event') type = val;
      else if (field === 'data') data.push(val);
    }
    if (!data.length) return;
    const raw = data.join('\n');
    let parsed;
    try { parsed = JSON.parse(raw); } catch (_) { parsed = { raw }; }
    onEvent(type, parsed || {});
  }
  return {
    push(chunk) {
      buf = (buf + chunk).replace(/\r\n?/g, '\n');
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, i);
        buf = buf.slice(i + 2);
        parseBlock(block);
      }
    },
    end() { if (buf.trim()) parseBlock(buf); buf = ''; },
  };
}

/* ───────────── 5. handleEvent：事件 → 状态 → 视图 ─────────────
 * 字段口径以 src/ragwriter/agent/handlers 实际发出的为准：
 *   section 编号：导语 0、正文 1..N、结尾 N+1；kind ∈ body|lede|ending|answer|notice
 *   verify：sentence（节内序号）、text、round（0=草稿，≥1=改写候选）、result、failed[]、checks[]、reason（「V3: …；V2: …」）
 *   patch：sentence、action（rewrite|delete）、text、original、reason
 * 返回 Promise 的分支（patch）会让后续事件排队等待，保证「红删除线保留 1.5 秒」可见。
 */
function handleEvent(turn, type, d) {
  d = d || {};
  turn.events += 1;
  switch (type) {
    case 'session': {
      if (d.session_id) app.sessionId = d.session_id;
      if (d.run_id) { turn.runId = d.run_id; app.runId = d.run_id; renderReplayBanner(); }
      return;
    }
    case 'intent': {
      turn.intent = {
        intent: d.intent, genre: d.genre, topic: d.topic, query: d.query, source: d.source,
        reason: d.reason, inherited: d.inherited, slots: d.slots || {},
        revise_target: d.revise_target || '', revise_action: d.revise_action || '',
        revise_instruction: d.revise_instruction || '', confirm_action: d.confirm_action || '',
      };
      if (d.genre) turn.genre = d.genre;
      renderIntentTag(turn);
      return;
    }
    case 'plan': {
      turn.plan = {
        title: d.title || '',
        genre: d.genre || turn.genre,
        outline: d.outline || (d.sections || []).map((s) => s.heading || s),
        subQuestions: d.sub_questions || [],
        insufficient: d.insufficient || [],
        groups: d.groups || null,
      };
      if (d.genre) turn.genre = d.genre;
      renderPlan(turn);
      return;
    }
    case 'confirm': {
      // 写前确认（10-23 子曰定）：本轮只出确认卡，不检索、不写作；用户点按钮或在追问栏回复
      turn.confirm = d;
      renderConfirm(turn);
      return;
    }
    case 'insufficient': {
      addInsufficient(turn, { topic: d.topic, signal: d.signal, section: d.section, nearest_pages: d.nearest_pages || [] });
      if (d.section != null && turn.sections.has(secKey('body', d.section))) markSectionInsufficient(turn, getSection(turn, d.section, 'body'));
      renderPlan(turn);
      return;
    }
    case 'retrieval': {
      for (const h of d.hits || []) if (h.sid) turn.sidInfo[h.sid] = Object.assign(turn.sidInfo[h.sid] || {}, { fact_id: h.fact_id, page_no: h.page_no, status: h.status });
      for (const s of d.snippets || []) turn.sidInfo[s.sid] = Object.assign(turn.sidInfo[s.sid] || {}, { fact_id: s.fact_id, page_no: s.page_no }, s.status ? { status: s.status } : {});
      turn.retrievals.push(d);
      renderRetrieval(turn);
      return;
    }
    case 'delta': {
      if (d.kind === 'notice') { turn.notice += d.text || ''; renderNotice(turn); return; }
      const sec = getSection(turn, d.section, d.kind);
      sec.draft += d.text || '';
      if (sec.status === 'idle') setSectionStatus(turn, sec, 'writing');
      paintTail(turn, sec);
      return;
    }
    case 'verify': {
      const sec = getSection(turn, d.section, d.kind);
      if (sec.status !== 'final') setSectionStatus(turn, sec, 'verifying');
      const round = d.round || 0;
      const pass = d.result === 'pass' || d.ok === true;
      const failed = d.failed || (d.check ? [d.check] : []);
      if (round === 0) {
        const s = addSentence(turn, sec, d.sentence, d.text);
        s.checks = d.checks || [];
        if (pass) { s.state = 'pass'; }
        else {
          s.state = 'fail'; s.failed = failed; s.reason = humanReason(d.reason);
          s.failAt = performance.now();
          turn.counts.blocked += 1;
        }
        paintSentence(turn, s);
      } else {
        const s = sec.sents.get(d.sentence);
        if (s) {
          s.attempts.push({ round, text: d.text, pass, reason: d.reason });
          if (s.state === 'fail') { s.note = pass ? `改写第 ${round} 轮已通过，等待替换…` : `改写第 ${round} 轮仍未通过：${humanReason(d.reason)}`; paintSentence(turn, s); }
        }
      }
      renderCounter(turn);
      renderSectionBadge(turn, sec);
      return;
    }
    case 'patch': {
      const sec = getSection(turn, d.section, d.kind);
      if (d.action === 'trim') {
        // 整块替换草稿（GAPS 行漏出后的清理、改写「短一点」的二次压缩），句子还没切出来，只换灰色草稿
        sec.draft = d.text || ''; sec.cursor = 0; paintTail(turn, sec);
        return;
      }
      let s = sec.sents.get(d.sentence);
      if (!s) s = addSentence(turn, sec, d.sentence, d.original || '');
      return applyPatch(turn, sec, s, d);
    }
    case 'section_final': {
      const sec = getSection(turn, d.section, d.kind);
      if (d.heading) sec.heading = d.heading;
      Object.entries(d.citations || {}).forEach(([sid, c]) => { turn.cites[sid] = c; turn.sidInfo[sid] = Object.assign(turn.sidInfo[sid] || {}, c); });
      if (d.insufficient) { markSectionInsufficient(turn, sec); return; }
      for (const fs of d.sentences || []) {
        let s = sec.sents.get(fs.idx);
        if (!s) s = addSentence(turn, sec, fs.idx, fs.text);
        if (s.state !== 'deleted') { s.text = fs.text; s.cites = fs.cites || sidsIn(fs.text); if (s.state !== 'rewritten') s.state = 'pass'; }
        s.final = true;
      }
      sec.cursor = sec.draft.length;   // 定稿以后端句子为准，残余草稿清掉
      sec.final = true;
      setSectionStatus(turn, sec, 'final');
      paintTail(turn, sec);
      repaintAll(turn);
      renderCounter(turn);
      return;
    }
    case 'error': {
      // rewrite / lede / ending / rewrite_query 失败后端会降级继续（改写失败按删除处理），只做提示
      const soft = ['rewrite', 'lede', 'ending', 'rewrite_query'].includes(d.stage);
      turn.errors.push(d);
      renderError(turn, d, !soft);
      return;
    }
    case 'done': {
      turn.done = d;
      if (d.run_id) { turn.runId = d.run_id; app.runId = d.run_id; renderReplayBanner(); }
      if (d.session_id) app.sessionId = d.session_id;
      if (d.genre) turn.genre = d.genre;
      if (d.injected) turn.injected = true;
      Object.entries(d.citations || {}).forEach(([sid, c]) => { turn.cites[sid] = c; turn.sidInfo[sid] = Object.assign(turn.sidInfo[sid] || {}, c); });
      finishTurn(turn);
      return;
    }
    default:
      return; // 未知事件忽略，保证后端加事件不会把前端搞崩
  }
}

/* ───────────── 6. 渲染 ───────────── */
function newTurn(userText, opts) {
  const id = 't' + (++app.seq);
  const root = el('article', 'turn');
  root.dataset.turn = id;
  root.innerHTML = `
    <div class="turn-user"><span class="turn-user-label">你</span><span class="turn-user-text">${esc(userText)}</span></div>
    <div class="turn-head">
      <div class="intent-tag" data-r="intent"><span class="tag tag-wait">识别意图…</span></div>
      <div class="inject-note" data-r="inject" hidden>演示注入：人为注入一处数字错误，用来展示校验闸（不是模型自然犯错）</div>
    </div>
    <div class="confirm-card plan" data-r="confirm" hidden></div>
    <div class="plan" data-r="plan" hidden></div>
    <details class="retrieval" data-r="retrieval" hidden><summary>检索过程</summary><div class="retrieval-body"></div></details>
    <div class="counter" data-r="counter" hidden></div>
    <div class="article" data-r="article">
      <h2 class="article-title" data-r="title" hidden></h2>
    </div>
    <div class="notice" data-r="notice" hidden></div>
    <div class="errors" data-r="errors"></div>
    <div class="report" data-r="report" hidden></div>
    <div class="boundary" data-r="boundary" hidden></div>
    <div class="jsonld" data-r="jsonld" hidden></div>
    <div class="turn-foot">
      <button class="btn btn-sm" data-r="copy" disabled title="校验完成后可复制，确保复制出去的每一句都已核实">复制全文</button>
      <span class="turn-meta" data-r="meta">生成中…</span>
    </div>`;
  const r = {};
  root.querySelectorAll('[data-r]').forEach((n) => { r[n.dataset.r] = n; });
  const turn = {
    id, root, r, userText, events: 0,
    genre: opts.genre || null, injected: !!opts.injected,
    intent: null, plan: null, runId: null, confirm: null,
    sections: new Map(), sidInfo: {}, cites: {}, insufficient: [], retrievals: [], errors: [],
    notice: '', done: null,
    counts: { blocked: 0, rewritten: 0, deleted: 0 },
    chain: Promise.resolve(),
  };
  if (turn.injected) r.inject.hidden = false;
  r.copy.addEventListener('click', () => copyTurn(turn));
  app.turns.set(id, turn);
  $('turns').appendChild(root);
  return turn;
}

/* 意图标签（S5） */
function intentLabel(turn) {
  const it = turn.intent || {};
  const done = turn.done;
  const kind = done && refusalKind(turn);
  if (kind === 'insufficient_all') return ['资料不足', 'tag-warn'];
  if (kind === 'out_of_scope' || it.intent === 'out_of_scope') return ['超出范围', 'tag-muted'];
  if (kind === 'clarify' || it.intent === 'clarify') return ['需要澄清', 'tag-info'];
  if (kind === 'unsupported' || it.intent === 'explain') return ['暂不支持', 'tag-muted'];
  if (it.intent === 'revise') return ['修改上一篇', 'tag-ok'];
  if (it.intent === 'confirm') return [{ modify: '修改待写主题', pick: '换成备选主题', cancel: '取消写作' }[it.confirm_action] || '确认写作', 'tag-info'];
  if (done && done.awaiting_confirmation) return ['写作 · ' + (GENRE_LABEL[turn.genre || it.genre] || 'Blog') + ' · 待确认', 'tag-info'];
  if (it.intent === 'qa') return ['问答', 'tag-ok'];
  if (it.intent === 'write') return ['写作 · ' + (GENRE_LABEL[turn.genre || it.genre] || 'Blog'), 'tag-ok'];
  return [it.intent || '—', 'tag-muted'];
}
function renderIntentTag(turn) {
  const it = turn.intent || {};
  const [label, cls] = intentLabel(turn);
  // 指定 = 写作台选了体裁与主题，没做意图识别；识别 = 模型理解的对话输入
  const prefix = it.source === 'selected' ? '指定' : it.source === 'model' ? '识别' : '意图';
  let html = `<span class="tag ${cls}">${prefix}：${esc(label)}</span>`;
  if (it.source && it.source !== 'selected' && it.source !== 'model') html += `<span class="tag tag-src" title="意图来源">${it.source === 'rule' ? '规则兜底' : it.source === 'forced' ? '固定意图' : esc(it.source)}</span>`;
  if (it.intent === 'revise' && it.revise_action) html += `<span class="tag tag-info" title="${esc(it.revise_instruction)}">修改：${esc(reviseLabel(it.revise_target, it.revise_action))}</span>`;
  const inh = it.inherited;
  const inhTopic = inh && (Array.isArray(inh) ? (inh.includes('topic') ? it.topic : null) : inh.topic);
  const inhGenre = inh && (Array.isArray(inh) ? (inh.includes('genre') ? it.genre : null) : inh.genre);
  if (inhTopic) html += `<span class="inherit">沿用主题：${esc(inhTopic)}</span>`;
  if (inhGenre) html += `<span class="inherit">沿用体裁：${esc(GENRE_LABEL[inhGenre] || inhGenre)}</span>`;
  if (it.intent === 'write' && !it.genre && !inhGenre) html += `<span class="inherit">默认按官网 Blog 写，可改为 FAQ / 品牌介绍 / 产品介绍</span>`;
  turn.r.intent.innerHTML = html;
}

const REVISE_ACTION_LABEL = { shorten: '缩短', expand: '扩写', tone: '换语气', rewrite: '重写', delete: '删除', add_section: '新增一节', retitle: '改标题' };
function reviseLabel(target, action) {
  let where = target && target.startsWith('section:') ? `第 ${target.split(':')[1]} 节`
    : ({ lede: '导语', ending: '结尾', title: '标题', all: '全文' }[target] || target || '全文');
  if (action === 'add_section') where = target && target.startsWith('section:') ? `第 ${target.split(':')[1]} 节之后` : '文末';
  return `${where} · ${REVISE_ACTION_LABEL[action] || action}`;
}

/* 写前确认卡：精确主题、体裁、侧重、要点、资料缺口、备选主题 + 确认 / 修改 */
function renderConfirm(turn) {
  const c = turn.confirm;
  if (!c) return;
  const genre = GENRE_LABEL[c.genre] || c.genre_label || c.genre || '';
  const li = (xs) => xs.map((x) => `<li>${esc(x)}</li>`).join('');
  let html = `<div class="confirm-head"><b>${c.valid === false ? '这个主题资料撑不起来' : '写之前确认一下'}</b></div>`;
  if (c.confirm_text) html += `<p class="confirm-text">${esc(c.confirm_text)}</p>`;
  html += `<div><span class="muted">主题：</span><b>${esc(c.topic || '')}</b> <span class="tag tag-muted">${esc(genre)}</span></div>`;
  if (c.angle) html += `<div><span class="muted">${c.genre === 'blog' ? '中心观点' : '侧重'}：</span>${esc(c.angle)}</div>`;
  if ((c.outline_points || []).length) html += `<div class="muted">打算写的要点：</div><ol class="plan-list">${li(c.outline_points)}</ol>`;
  if ((c.premise_gaps || []).length) html += (c.premise_gaps || []).map((g) => `<div class="plan-insuf"><b>资料缺口</b>${esc(g)}<span class="muted">（这部分品宣里没有，文中不会写）</span></div>`).join('');
  if (c.valid === false && c.reason) html += `<div class="plan-insuf"><b>原因</b>${esc(c.reason)}</div>`;
  if ((c.alternatives || []).length) html += `<div class="refusal-sugg"><span>备选主题：</span>${c.alternatives.map((a) => `<button type="button" class="btn btn-sm confirm-pick" data-topic="${esc(a)}">${esc(a)}</button>`).join('')}</div>`;
  html += `<div class="refusal-sugg">${c.valid === false ? '' : '<button type="button" class="btn btn-primary btn-sm confirm-accept">确认，开始写作</button>'}<button type="button" class="btn btn-sm confirm-modify">修改主题</button></div>`;
  turn.r.confirm.innerHTML = html;
  turn.r.confirm.hidden = false;
  const lock = () => turn.r.confirm.querySelectorAll('button').forEach((b) => { b.disabled = true; });
  const acc = turn.r.confirm.querySelector('.confirm-accept');
  if (acc) acc.addEventListener('click', () => { if (app.busy) return; lock(); send('确认', { genre: c.genre, options: { confirm: 'accept' } }); });
  turn.r.confirm.querySelectorAll('.confirm-pick').forEach((b) => b.addEventListener('click', () => {
    if (app.busy) return; lock(); send(`就写：${b.dataset.topic}`, { genre: c.genre, options: { confirm: 'pick', topic: b.dataset.topic } });
  }));
  turn.r.confirm.querySelector('.confirm-modify').addEventListener('click', () => {
    const inp = $('followup-input');
    $('followup').hidden = false;
    if (!inp.value.trim()) inp.value = '主题改为：';
    inp.focus();
    inp.setSelectionRange(inp.value.length, inp.value.length);
  });
}

/* 大纲与资料不足（plan 事件里先声明哪部分资料没有） */
function addInsufficient(turn, item) {
  if (!item.topic) return;
  const ex = turn.insufficient.find((x) => x.topic === item.topic);
  if (ex) { if (item.nearest_pages && item.nearest_pages.length) ex.nearest_pages = item.nearest_pages; return; }
  turn.insufficient.push(item);
}
function renderPlan(turn) {
  const p = turn.plan;
  if (!p) return;
  (p.insufficient || []).forEach((t) => addInsufficient(turn, { topic: t, signal: 'planner', nearest_pages: [] }));
  const isFaq = (turn.genre || p.genre) === 'faq';
  const items = (p.outline || []).map((h, i) => `<li>${isFaq ? '<b class="q-mark">Q</b> ' : ''}${esc(h)}${p.subQuestions[i] && p.subQuestions[i] !== h ? `<span class="subq">子问题：${esc(p.subQuestions[i])}</span>` : ''}</li>`).join('');
  const insuf = turn.insufficient.map((x) => `<div class="plan-insuf"><b>资料不足</b>${esc(x.topic)}<span class="muted">（不写入正文，见文末资料边界）</span></div>`).join('');
  turn.r.plan.innerHTML = `<details open><summary>大纲 · ${(p.outline || []).length} ${isFaq ? '问' : '节'}</summary><ol class="plan-list">${items}</ol></details>${insuf}`;
  turn.r.plan.hidden = false;
  if (p.title) { turn.r.title.textContent = p.title; turn.r.title.hidden = false; }
}

function renderRetrieval(turn) {
  const rows = turn.retrievals.map((d) => {
    const pages = [];
    (d.hits || []).forEach((h) => {
      const key = h.page_no;
      if (!pages.some((x) => x.p === key)) pages.push({ p: key, bm: h.bm25_rank, de: h.dense_rank, st: h.status });
    });
    const chips = pages.slice(0, 8).map((x) => `<button type="button" class="page-link" data-page="${x.p}" title="BM25 #${x.bm ?? '—'} · 稠密 #${x.de ?? '—'}">p${x.p}${x.st === 'image_structural' ? '·图' : ''}</button>`).join(' ');
    const label = d.kind === 'answer' ? '问答' : d.section != null ? `第 ${d.section} 节` : '';
    return `<div class="retr-row"><span class="retr-sec">${esc(label)}</span><span class="retr-q">${esc(d.query || '')}</span>
      <span class="retr-ok ${d.sufficient ? 'ok' : 'bad'}">${d.sufficient ? '资料充分' : '资料不足'}</span>
      ${d.dense_used === false ? '<span class="muted">（仅 BM25）</span>' : ''}<span class="retr-pages">${chips}</span></div>`;
  }).join('');
  turn.r.retrieval.querySelector('.retrieval-body').innerHTML = rows;
  turn.r.retrieval.hidden = false;
}

/* 小节 */
function secKey(kind, no) {
  if (kind === 'lede') return 'lede';
  if (kind === 'ending') return 'ending';
  return 'b' + no;
}
function secOrder(kind, no) {
  if (kind === 'lede') return -1;
  if (kind === 'ending') return 1e6;
  return Number(no) || 0;
}
function getSection(turn, no, kind) {
  kind = kind || 'body';
  const key = secKey(kind, no);
  let sec = turn.sections.get(key);
  if (sec) return sec;
  const order = secOrder(kind, no);
  const isFaq = turn.genre === 'faq';
  let heading = '';
  if (kind === 'body' && turn.plan && turn.plan.outline) heading = turn.plan.outline[(Number(no) || 1) - 1] || '';
  const node = el('section', `sec sec-${kind}${isFaq && kind === 'body' ? ' sec-faq' : ''}`);
  node.dataset.order = order;
  const label = kind === 'lede' ? '导语' : kind === 'ending' ? '结尾' : kind === 'answer' ? '回答' : '';
  node.innerHTML = `
    <div class="sec-head">
      ${kind === 'body' ? `<h3 class="sec-title">${isFaq ? '<b class="q-mark">Q</b>' : ''}<span data-h></span></h3>` : `<span class="sec-label">${label}</span>`}
      <span class="badge" data-b></span>
    </div>
    <p class="sec-body">${isFaq && kind === 'body' ? '<b class="a-mark">A</b>' : ''}<span class="s-tail"></span></p>`;
  sec = {
    key, no, kind, order, heading, node,
    headEl: node.querySelector('[data-h]'), badgeEl: node.querySelector('[data-b]'),
    bodyEl: node.querySelector('.sec-body'), tailEl: node.querySelector('.s-tail'),
    draft: '', cursor: 0, sents: new Map(), status: 'idle', final: false, insufficient: false,
  };
  if (sec.headEl) sec.headEl.textContent = heading || `第 ${no} 节`;
  if (kind === 'body' && turn.insufficient.some((x) => x.section === no)) setTimeout(() => markSectionInsufficient(turn, sec), 0);
  // FAQ 分组标题（plan.groups 形如 [{name, questions:[节号或问题文本]}]，没有就不分组）
  if (isFaq && kind === 'body' && turn.plan && Array.isArray(turn.plan.groups)) {
    const g = turn.plan.groups.find((x) => (x.questions || [])[0] === no || (x.questions || [])[0] === heading);
    if (g) node.insertBefore(el('div', 'faq-group', esc(g.name || g.title || '')), node.firstChild);
  }
  // 按 order 插入：导语在最前，结尾在最后（导语是在正文之后才流式写的）
  const article = turn.r.article;
  const after = Array.from(article.querySelectorAll('section.sec')).find((n) => Number(n.dataset.order) > order);
  article.insertBefore(node, after || null);
  turn.sections.set(key, sec);
  turn.r.counter.hidden = false;
  renderCounter(turn);
  return sec;
}
function setSectionStatus(turn, sec, status) { sec.status = status; renderSectionBadge(turn, sec); }
function renderSectionBadge(turn, sec) {
  const all = Array.from(sec.sents.values());
  const nRew = all.filter((s) => s.state === 'rewritten').length;
  const nDel = all.filter((s) => s.state === 'deleted').length;
  let text = '', cls = 'badge';
  if (sec.insufficient) { text = '资料不足 · 未写作'; cls += ' badge-warn'; }
  else if (sec.status === 'writing') { text = '写作中…'; cls += ' badge-run'; }
  else if (sec.status === 'verifying') { text = `校验中… 已核 ${all.length} 句`; cls += ' badge-run'; }
  else if (sec.status === 'final') { text = `✓ 已定稿 · ${all.filter((s) => s.state !== 'deleted').length} 句已核 · 改写 ${nRew} · 删除 ${nDel}`; cls += ' badge-ok'; }
  sec.badgeEl.className = cls;
  sec.badgeEl.textContent = text;
}
function markSectionInsufficient(turn, sec) {
  sec.insufficient = true; sec.final = true; sec.status = 'final';
  sec.node.classList.add('sec-insuf');
  sec.bodyEl.innerHTML = '<span class="insuf-note">本节品宣资料不足，未写作（不进入发布稿），见文末「资料边界」。</span>';
  sec.tailEl = el('span'); // 后续误到的 delta 不再显示
  renderSectionBadge(turn, sec);
}

/* 句子：草稿尾巴（灰）+ 已切出的句子（按 verify 到达） */
function addSentence(turn, sec, idx, text) {
  let s = sec.sents.get(idx);
  if (s) return s;
  text = text || '';
  const pos = text ? sec.draft.indexOf(text.trim(), sec.cursor) : -1;
  if (pos >= 0) sec.cursor = pos + text.trim().length;
  else sec.cursor = Math.min(sec.draft.length, sec.cursor + text.length);
  s = { idx, text, cites: sidsIn(text), state: 'draft', reason: '', note: '', failed: [], checks: [], attempts: [], failAt: 0, flash: false, original: '' };
  s.el = el('span', 's');
  // 插入到同节内 idx 更大的句子之前，否则放在草稿尾巴之前
  const later = Array.from(sec.sents.values()).filter((x) => x.idx > idx).sort((a, b) => a.idx - b.idx)[0];
  sec.bodyEl.insertBefore(s.el, later ? later.el : sec.tailEl);
  sec.sents.set(idx, s);
  paintTail(turn, sec);
  return s;
}
function paintTail(turn, sec) {
  if (sec.insufficient) return;
  let rest = sec.draft.slice(sec.cursor);
  rest = rest.replace(/\[[^\]]*$/, ''); // 半截引用「[S1」先不显示
  sec.tailEl.className = rest.trim() ? 's-tail s-draft' : 's-tail';
  sec.tailEl.innerHTML = rest.trim() ? renderCites(turn, rest, false) : '';
}
function paintSentence(turn, s) {
  const cls = ['s'];
  let html;
  switch (s.state) {
    case 'fail':
      cls.push('s-fail');
      html = `<del>${renderCites(turn, s.text, false)}</del><span class="s-reason">被拦截：${esc(s.reason)}${s.note ? ` · ${esc(s.note)}` : ''}</span>`;
      break;
    case 'deleted':
      cls.push('s-del');
      html = esc(DELETED_PLACEHOLDER);
      break;
    case 'rewritten':
      cls.push('s-pass', 's-rew');
      if (s.flash) cls.push('s-flash');
      html = renderCites(turn, s.text, true);
      break;
    case 'pass':
      cls.push('s-pass');
      html = renderCites(turn, s.text, true);
      break;
    default:
      cls.push('s-draft');
      html = renderCites(turn, s.text, false);
  }
  if (s.out) cls.push('s-out');
  s.el.className = cls.join(' ');
  s.el.innerHTML = html;
  if (s.state === 'rewritten' && s.original) s.el.title = '改写前：' + stripCites(s.original);
}
function repaintAll(turn) {
  for (const sec of turn.sections.values()) { for (const s of sec.sents.values()) paintSentence(turn, s); paintTail(turn, sec); }
}
async function applyPatch(turn, sec, s, d) {
  if (s.state === 'fail') {
    const wait = CFG.holdMs - (performance.now() - s.failAt);
    if (wait > 0) await sleep(wait);
    s.out = true; paintSentence(turn, s);
    await sleep(250);
    s.out = false;
  }
  if (d.action === 'delete') {
    s.state = 'deleted';
    s.original = d.original || s.text;
    s.deleteReason = humanReason(d.reason || s.reason);
    turn.counts.deleted += 1;
  } else {
    s.original = d.original || s.text;
    s.text = d.text || s.text;
    s.cites = sidsIn(s.text);
    s.state = 'rewritten';
    s.flash = true;
    turn.counts.rewritten += 1;
    setTimeout(() => { s.flash = false; s.el.classList.remove('s-flash'); }, CFG.flashMs);
  }
  paintSentence(turn, s);
  renderCounter(turn);
  renderSectionBadge(turn, sec);
}

/* 引用角标：[S4] → p9；image_structural → p5·图（S3） */
function citeInfo(turn, sid) { return turn.cites[sid] || turn.sidInfo[sid] || null; }
function renderCites(turn, text, solid) {
  let out = '';
  let last = 0;
  const seen = new Set(); // 同一句里 [S1][S2] 都是 p9 时只出一个角标
  String(text).replace(CITE_RE, (m, grp, off) => {
    out += esc(text.slice(last, off));
    last = off + m.length;
    const chips = (grp.match(/S\d+/g) || []).map((sid) => {
      const c = citeInfo(turn, sid);
      const label = c && c.page_no != null ? `p${c.page_no}${c.status === 'image_structural' ? '·图' : ''}` : sid;
      if (seen.has(label)) return '';
      seen.add(label);
      return `<button type="button" class="cite${solid ? ' cite-solid' : ''}${c && c.status === 'image_structural' ? ' cite-img' : ''}" data-turn="${turn.id}" data-sid="${sid}" title="${esc(c && c.text ? c.text : '点击查看原页')}">${esc(label)}</button>`;
    }).join('');
    if (chips) out += `<sup class="cites">${chips}</sup>`;
    return m;
  });
  out += esc(text.slice(last));
  return out;
}

/* 计数器（M3）：事实句 / 带引用 / 拦截 / 改写 / 删除 */
function liveStats(turn) {
  let kept = 0, cited = 0;
  for (const sec of turn.sections.values()) for (const s of sec.sents.values()) {
    if (s.state === 'pass' || s.state === 'rewritten') { kept += 1; if (sidsIn(s.text).length) cited += 1; }
  }
  return { kept, cited };
}
function renderCounter(turn) {
  const { kept, cited } = liveStats(turn);
  const rep = turn.done && turn.done.report;
  const fact = rep && rep.factual_sentences != null ? rep.factual_sentences : kept;
  const cit = rep && rep.cited_sentences != null ? rep.cited_sentences : cited;
  const c = turn.counts;
  turn.r.counter.innerHTML = `
    <span class="cnt"><b>${fact}</b>事实句</span><span class="cnt"><b>${cit}</b>带引用</span>
    <span class="cnt cnt-block${c.blocked ? ' hot' : ''}"><b>${c.blocked}</b>拦截</span>
    <span class="cnt"><b>${c.rewritten}</b>改写</span><span class="cnt"><b>${c.deleted}</b>删除</span>
    <span class="cnt-state">${turn.done ? '校验完成' : '生成与校验中…'}</span>`;
}

function renderNotice(turn) {
  turn.r.notice.hidden = false;
  turn.r.notice.textContent = turn.notice;
}
function renderError(turn, d, fatal) {
  let msg;
  if (d.reason === 'replay_unmatched' || d.code === 'replay_unmatched') {
    msg = '回放模式只能播放已录制的场景，请从下拉列表中选择一个录制场景。';
    fatal = true;
  } else {
    msg = `${fatal ? '出错了' : '提示'}（阶段：${esc(d.stage || '未知')}）：${esc(d.reason || d.message || '未知错误')}${d.fallback ? ' · 已切换备用模型' : ''}`;
  }
  turn.r.errors.appendChild(el('div', fatal ? 'err err-fatal' : 'err', msg));
}

/* 拒答三类：out_of_scope / insufficient_all / clarify（以及暂不支持） */
function refusalKind(turn) {
  const d = turn.done || {};
  if (!d.refused) return null;
  if (d.refusal_kind) return d.refusal_kind;
  const it = (turn.intent && turn.intent.intent) || d.intent;
  if (it === 'out_of_scope') return 'out_of_scope';
  if (it === 'clarify') return 'clarify';
  if (it === 'explain' || d.unsupported) return 'unsupported';
  return 'insufficient_all';
}
function renderRefusal(turn) {
  const d = turn.done;
  const kind = refusalKind(turn);
  const text = d.final_text || turn.notice || '';
  const near = d.nearest_pages || [];
  const sugg = d.suggestions || [];
  const titles = {
    insufficient_all: '资料不足，不做推测',
    out_of_scope: '超出工作范围',
    clarify: '需要补充信息',
    unsupported: '本版本暂不支持',
  };
  let html = `<div class="refusal-title">${esc(titles[kind] || '未生成内容')}</div><div class="refusal-text">${esc(text)}</div>`;
  if (kind === 'out_of_scope' && !sugg.length) {
    html += '<div class="refusal-sub">我只根据万悉科技品宣 PDF 写官网内容（Blog / FAQ / 品牌介绍 / 产品介绍），或回答资料里的问题。</div>';
  }
  if (near.length) {
    html += `<div class="refusal-near"><span>与之最接近的资料：</span>${near.map((p) => `<button type="button" class="btn btn-sm page-link" data-page="${p}">第 ${p} 页</button>`).join('')}</div>`;
  }
  const list = sugg.length ? sugg : (kind === 'out_of_scope' ? ['写一篇官网 Blog：什么是 LLM-原生 GEO？'] : []);
  if (list.length) {
    html += `<div class="refusal-sugg"><span>你可以试试：</span>${list.map((s) => `<button type="button" class="btn btn-sm sugg" data-msg="${esc(topicText(s))}" data-kind="${kind}">${esc(topicText(s))}</button>`).join('')}</div>`;
  }
  turn.r.notice.className = `notice refusal refusal-${kind}`;
  turn.r.notice.innerHTML = html;
  turn.r.notice.hidden = false;
}

/* 校验摘要（S4）+ 完整报告 */
function renderReport(turn) {
  const d = turn.done;
  const rep = d.report || {};
  const boundary = boundaryItems(turn);
  const nums = rep.numbers || [];
  const numOk = nums.length - (rep.numbers_inconsistent != null ? rep.numbers_inconsistent : nums.filter((x) => !x.consistent).length);
  const rew = rep.rewrites || [];
  const del = rep.deletions || [];
  const { kept, cited } = liveStats(turn);
  const fact = rep.factual_sentences != null ? rep.factual_sentences : kept;
  const cit = rep.cited_sentences != null ? rep.cited_sentences : cited;
  const blocked = turn.counts.blocked;
  const summary = `事实句 ${fact} · 带引用 ${cit} · 数字一致 ${numOk}/${nums.length} · 拦截 ${blocked} · 删除 ${del.length || turn.counts.deleted} · 资料边界 ${boundary.length}`;
  const gates = rep.gate_blocks ? Object.entries(rep.gate_blocks).map(([k, v]) => `<span class="gate${v ? ' hot' : ''}">${k} ${esc(CHECK_LABEL[k] || '')}：${v}</span>`).join('') : '';
  const rewRows = rew.map((x) => `<tr><td>${esc(stripCites(x.original))}</td><td>${esc((x.failed_checks || []).join(', '))}</td><td>${esc(humanReason(x.reason))}</td><td>${esc(stripCites(x.rewritten))}</td></tr>`).join('');
  const delRows = del.map((x) => `<tr><td>${esc(stripCites(x.original))}</td><td>${esc((x.failed_checks || []).join(', ') || (String(x.reason || '').match(/V\d/g) || []).join(', '))}</td><td>${esc(humanReason(x.reason))}</td></tr>`).join('');
  const numRows = nums.map((x) => `<tr><td>${esc(x.number)}</td><td>${(x.pages || []).map((p) => `p${p}`).join(' ')}</td><td>${esc(x.snippet || '')}</td><td class="${x.consistent ? 'ok' : 'bad'}">${x.consistent ? '一致' : '不一致'}</td></tr>`).join('');
  const insRows = (rep.insufficient || turn.insufficient).map((x) => `<li>${esc(x.topic)} <span class="muted">（${esc(x.signal === 'planner' ? '规划阶段声明' : x.signal || '')}）</span></li>`).join('');
  turn.r.report.innerHTML = `
    <details>
      <summary><span class="report-label">校验摘要</span><span class="report-line">${esc(summary)}</span><span class="report-more">展开完整校验报告</span></summary>
      <div class="report-body">
        ${gates ? `<div class="gates"><b>各闸失败次数</b>（含改写过程中的每次失败）${gates}</div>` : ''}
        <h4>改写记录（${rew.length}）</h4>
        ${rew.length ? `<div class="tbl-wrap"><table><thead><tr><th>原句</th><th>检查</th><th>原因</th><th>改写后</th></tr></thead><tbody>${rewRows}</tbody></table></div>` : '<p class="muted">无</p>'}
        <h4>删除记录（${del.length}）</h4>
        ${del.length ? `<div class="tbl-wrap"><table><thead><tr><th>原句</th><th>检查</th><th>原因</th></tr></thead><tbody>${delRows}</tbody></table></div>` : '<p class="muted">无</p>'}
        <h4>数字清单（${numOk}/${nums.length} 一致）</h4>
        ${nums.length ? `<div class="tbl-wrap"><table><thead><tr><th>数字</th><th>引用页</th><th>原文片段</th><th>结论</th></tr></thead><tbody>${numRows}</tbody></table></div>` : '<p class="muted">无</p>'}
        <h4>资料不足</h4>${insRows ? `<ul>${insRows}</ul>` : '<p class="muted">无</p>'}
      </div>
    </details>`;
  turn.r.report.hidden = false;
}

/* 资料边界卡（M1）+ 出处说明 + 参考页码 */
function boundaryItems(turn) {
  const d = turn.done || {};
  if (Array.isArray(d.boundary)) return d.boundary;
  return turn.insufficient.filter((x) => x.signal === 'planner' || x.section != null);
}
function renderBoundary(turn) {
  const items = boundaryItems(turn);
  const pages = Array.from(new Set(Object.values(turn.cites).map((c) => c.page_no).filter((p) => p != null))).sort((a, b) => a - b);
  let html = '';
  if (items.length) {
    html += `<div class="bd-card">
      <div class="bd-title">资料边界</div>
      <div class="bd-lead">本文未涉及（品宣资料中没有）：</div>
      <ul class="bd-list">${items.map((x) => `<li><span>${esc(x.topic)}</span>${(x.nearest_pages || []).length ? `<span class="bd-near">最近页：${x.nearest_pages.map((p) => `<button type="button" class="page-link" data-page="${p}">p${p}</button>`).join(' ')}</span>` : ''}</li>`).join('')}</ul>
      <div class="bd-hint">本文不做此类论断。如需补充，可提供行业报告或客户调研数据后重新生成。</div>
    </div>`;
  }
  if (pages.length) {
    html += `<div class="bd-src">本文数据均引自万悉科技品宣资料，出处见文中页码标注。</div>
      <div class="bd-pages"><span>参考页码：</span>${pages.map((p) => `<button type="button" class="page-link" data-page="${p}">p${p}</button>`).join(' ')}</div>`;
  }
  turn.r.boundary.innerHTML = html;
  turn.r.boundary.hidden = !html;
}

function renderJsonLd(turn) {
  const j = turn.done.jsonld;
  if (!j) return;
  const txt = typeof j === 'string' ? j : JSON.stringify(j, null, 2);
  turn.r.jsonld.innerHTML = `<details><summary>查看 JSON-LD（FAQPage 结构化数据，可直接放进官网页面）</summary>
    <div class="jsonld-actions"><button type="button" class="btn btn-sm" data-copy-jsonld>复制 JSON-LD</button></div>
    <pre><code>${esc(txt)}</code></pre></details>`;
  turn.r.jsonld.querySelector('[data-copy-jsonld]').addEventListener('click', (e) => copyText(`<script type="application/ld+json">\n${txt}\n</script>`, e.target));
  turn.r.jsonld.hidden = false;
}

function renderMeta(turn) {
  const d = turn.done || {};
  const u = d.usage || {};
  const elapsed = d.elapsed_ms != null ? d.elapsed_ms : u.elapsed_ms;
  const sm = d.stream_mode === 'native' ? '原生流式' : d.stream_mode === 'simulated' ? '模拟流式' : d.stream_mode === 'none' ? '未调用模型' : (d.stream_mode || '—');
  const parts = [`流式：${sm}`, `输入 ${fmtNum(u.prompt_tokens)} / 输出 ${fmtNum(u.completion_tokens)} tokens`, `耗时 ${fmtMs(elapsed)}`];
  const cost = d.cost_cny != null ? d.cost_cny : u.cost_cny;
  if (cost != null) parts.push(`约 ${Number(cost).toFixed(2)} 元`);
  if (d.replay) parts.push('回放');
  if (turn.runId) parts.push(`run ${turn.runId}`);
  turn.r.meta.textContent = parts.join(' · ');
}

function finishTurn(turn) {
  const d = turn.done;
  // 残余的未确认草稿统一收口
  for (const sec of turn.sections.values()) { if (!sec.final && !sec.insufficient) { sec.cursor = sec.draft.length; setSectionStatus(turn, sec, 'final'); } paintTail(turn, sec); }
  repaintAll(turn);
  renderIntentTag(turn);
  renderCounter(turn);
  if (d.refused) {
    renderRefusal(turn);
    turn.r.counter.hidden = turn.sections.size === 0;
  } else {
    turn.r.notice.hidden = !turn.notice;
    if (turn.sections.size) renderReport(turn);
  }
  renderBoundary(turn);
  renderJsonLd(turn);
  renderMeta(turn);
  turn.root.classList.add('turn-done');
  const canCopy = !d.refused && turn.sections.size > 0 && !d.error;
  turn.r.copy.disabled = !canCopy;
  turn.r.copy.title = canCopy ? '复制定稿（不含被删除的句子与资料不足小节）' : '本轮没有可复制的定稿';
}

/* 复制：只复制定稿句，引用写成 [p9]，附出处说明与参考页码 */
function buildCopyText(turn) {
  const lines = [];
  const plain = (t) => t.replace(CITE_RE, (_, grp) => {
    const labels = Array.from(new Set((grp.match(/S\d+/g) || []).map((sid) => { const c = citeInfo(turn, sid); return c && c.page_no != null ? `p${c.page_no}` : sid; })));
    return `[${labels.join(' ')}]`;
  });
  const sentencesOf = (sec) => Array.from(sec.sents.values()).sort((a, b) => a.idx - b.idx).filter((s) => s.state === 'pass' || s.state === 'rewritten').map((s) => plain(s.text)).join('');
  const isFaq = turn.genre === 'faq';
  if (turn.plan && turn.plan.title) lines.push(`# ${turn.plan.title}`, '');
  const secs = Array.from(turn.sections.values()).sort((a, b) => a.order - b.order);
  for (const sec of secs) {
    if (sec.insufficient) continue;
    const body = sentencesOf(sec);
    if (!body) continue;
    if (sec.kind === 'body') lines.push(isFaq ? `Q：${sec.heading}` : `## ${sec.heading}`, '');
    lines.push(isFaq && sec.kind === 'body' ? `A：${body}` : body, '');
  }
  const pages = Array.from(new Set(Object.values(turn.cites).map((c) => c.page_no))).filter((p) => p != null).sort((a, b) => a - b);
  if (pages.length) lines.push('本文数据均引自万悉科技品宣资料，出处见文中页码标注。', `参考页码：${pages.map((p) => 'p' + p).join(' ')}`);
  return lines.join('\n').trim();
}
async function copyText(text, btn) {
  try { await navigator.clipboard.writeText(text); }
  catch (_) {
    const ta = el('textarea'); ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) { /* 忽略 */ }
    ta.remove();
  }
  if (btn) { const old = btn.textContent; btn.textContent = '已复制'; setTimeout(() => { btn.textContent = old; }, 1500); }
}
function copyTurn(turn) { if (turn.done) copyText(buildCopyText(turn), turn.r.copy); }

/* ───────────── 7. 页面面板 ───────────── */
function matchBlocks(blocks, factText) {
  const norm = (s) => String(s || '').replace(/\s+/g, '').toLowerCase();
  const nf = norm(factText);
  if (!nf) return [];
  const parts = String(factText).split(/[：:（）()，,、；;·\n“”"「」]/).map(norm).filter((p) => p.length >= 2 && !/^\d+$/.test(p));
  return blocks.filter((b) => {
    const bn = norm(b.text);
    if (bn.length < 2) return false;
    return nf.includes(bn) || bn.includes(nf) || parts.some((p) => p.length >= 3 && bn.includes(p));
  });
}
function openPanel(title, html) {
  $('panel-title').textContent = title;
  $('panel-body').innerHTML = html;
  $('panel').hidden = false;
  document.body.classList.add('panel-open');
}
function closePanel() { $('panel').hidden = true; document.body.classList.remove('panel-open'); }
function thumbHTML(page, boxes, wholeImage) {
  const src = page.image_url || `/build/pages/p${String(page.page_no).padStart(2, '0')}.png`;
  const W = page.width || CFG.pageW, H = page.height || CFG.pageH;
  const rects = boxes.map((b) => {
    const l = Math.max(0, b.x0 / W * 100), t = Math.max(0, b.y0 / H * 100);
    const w = Math.min(100 - l, (b.x1 - b.x0) / W * 100), h = Math.min(100 - t, (b.y1 - b.y0) / H * 100);
    return `<span class="hl" style="left:${l}%;top:${t}%;width:${w}%;height:${h}%"></span>`;
  }).join('');
  return `<div class="thumb${wholeImage ? ' thumb-whole' : ''}" style="aspect-ratio:${W}/${H}"><img src="${esc(src)}" alt="第 ${page.page_no} 页" loading="lazy" onerror="this.parentNode.classList.add('thumb-missing')">${rects}<span class="thumb-missing-text">页面图未取到</span></div>`;
}
async function openCite(turn, sid) {
  const c = citeInfo(turn, sid) || {};
  const pn = c.page_no;
  const isImg = c.status === 'image_structural';
  const head = `<div class="pn-meta"><span class="tag ${isImg ? 'tag-info' : 'tag-ok'}">${isImg ? '流程图结构关系' : '文字层原文'}</span><span class="muted">${esc(sid)}${c.fact_id ? ' · ' + esc(c.fact_id) : ''}</span></div>
    <blockquote class="pn-quote">${c.text ? esc(c.text) : '<span class="muted">（原文片段待定稿后提供）</span>'}</blockquote>
    ${isImg ? '<div class="pn-note">来自流程图关系，文字层只有节点名。此依据取自页面图中的箭头/结构关系，文字层没有对应整句原文，已经过人工核对，故高亮整张图。</div>' : ''}`;
  if (pn == null) { openPanel(`${sid} · 依据`, head); return; }
  openPanel(`第 ${pn} 页 · 引用依据`, head + '<div class="muted">加载页面…</div>');
  try {
    const page = await fetchPage(pn);
    const boxes = isImg ? [] : matchBlocks(page.blocks || [], c.text || '');
    openPanel(`第 ${pn} 页 · ${page.title || '引用依据'}`, head + thumbHTML(page, boxes, isImg) +
      `${!isImg && !boxes.length && c.text ? '<div class="muted small">未能在文字层定位到对应文本块，请对照下方全文。</div>' : ''}
       <details class="pn-text"><summary>本页文字层全文</summary><pre>${esc(page.text || '')}</pre></details>`);
  } catch (e) {
    openPanel(`第 ${pn} 页 · 引用依据`, head + `<div class="err">页面数据加载失败：${esc(e.message)}</div>`);
  }
}
async function openPage(pn) {
  openPanel(`第 ${pn} 页`, '<div class="muted">加载页面…</div>');
  try {
    const page = await fetchPage(pn);
    openPanel(`第 ${pn} 页 · ${page.title || ''}`, thumbHTML(page, [], false) + `<details class="pn-text" open><summary>本页文字层全文</summary><pre>${esc(page.text || '')}</pre></details>`);
  } catch (e) {
    openPanel(`第 ${pn} 页`, `<div class="err">页面数据加载失败：${esc(e.message)}</div>`);
  }
}

/* ───────────── 8. 写作台 / 追问 / 回放 / 注入 ───────────── */
function isReplay() { return app.mode && app.mode.mode === 'replay'; }
function selectedGenre() { const r = document.querySelector('input[name="genre"]:checked'); return r ? r.value : 'blog'; }

async function send(message, opts) {
  opts = opts || {};
  if (app.busy || !message.trim()) return;
  app.busy = true;
  setBusy(true);
  $('empty').hidden = true;
  $('followup').hidden = false;
  const inject = $('inject').checked && !isReplay();
  const options = Object.assign({}, opts.options || {});
  if (inject) options.inject_fault = 'v3_swap';
  const body = { message };
  if (app.sessionId) body.session_id = app.sessionId;
  if (Object.keys(options).length) body.options = options;
  const turn = newTurn(message, { genre: opts.genre, injected: inject });
  turn.root.scrollIntoView({ behavior: 'smooth', block: 'start' });
  app.abort = new AbortController();
  try {
    await streamChat(body, (type, data) => { turn.chain = turn.chain.then(() => handleEvent(turn, type, data)).catch((e) => console.error('handleEvent', type, e)); }, app.abort.signal);
    await turn.chain;
    if (!turn.done) {
      renderError(turn, { stage: 'stream', reason: turn.events ? '事件流在 done 之前结束（连接中断或后端异常）' : '没有收到任何事件' }, true);
      turn.r.meta.textContent = '未完成';
    }
  } catch (e) {
    await turn.chain;
    if (e.name !== 'AbortError') renderError(turn, { stage: 'network', reason: `无法连接后端：${e.message}` }, true);
    turn.r.meta.textContent = '未完成';
  } finally {
    app.busy = false;
    app.abort = null;
    setBusy(false);
  }
}
function setBusy(b) {
  $('gen-btn').disabled = b;
  $('gen-btn').textContent = b ? '生成中…' : '生成';
  document.querySelectorAll('#followup button[type="submit"]').forEach((x) => { x.disabled = b; });
}

function deskSubmit(e) {
  if (e) e.preventDefault();
  const genre = selectedGenre();
  if (isReplay()) {
    const v = $('topic-select').value;
    if (!v) return;
    const item = JSON.parse(v);
    send(typeof item === 'string' ? item : (item.message || item.topic || item.label), { genre: item.genre || genre });
    return;
  }
  const topic = $('topic').value.trim();
  if (!topic) { $('desk-hint').textContent = '请输入写作主题。'; $('topic').focus(); return; }
  $('desk-hint').textContent = '';
  const options = { genre, topic };
  if (CFG.forceIntent) options.force_intent = { intent: 'write', genre, topic };
  send(`${GENRE_VERB[genre]}：${topic}`, { genre, options });
}
function followupSubmit(e) {
  e.preventDefault();
  if (isReplay()) {
    const v = $('followup-select').value;
    if (!v) return;
    const item = JSON.parse(v);
    send(typeof item === 'string' ? item : (item.message || item.topic || item.label), { genre: item.genre });
    return;
  }
  const v = $('followup-input').value.trim();
  if (!v) return;
  $('followup-input').value = '';
  send(v, {});
}
function newSession() {
  if (app.abort) app.abort.abort();
  app.sessionId = null;
  app.turns.clear();
  $('turns').innerHTML = '';
  $('followup').hidden = true;
  $('empty').hidden = false;
  closePanel();
}

function renderSuggestions(list) {
  const box = $('suggestions');
  box.innerHTML = '';
  list.slice(0, 3).forEach((s, i) => {
    const t = topicText(s);
    const b = el('button', 'sugg-card', `<span class="sugg-no">${i + 1}</span><span>${esc(t)}</span>${s && s.pages ? `<span class="muted small">依据 ${esc(s.pages)}</span>` : ''}`);
    b.type = 'button';
    b.addEventListener('click', () => {
      if (isReplay()) return;
      $('topic').value = t;
      deskSubmit();
    });
    box.appendChild(b);
  });
}

function renderReplayBanner() {
  const m = app.mode;
  const b = $('replay-banner');
  if (!m || m.mode !== 'replay') { b.hidden = true; return; }
  const run = app.runId || m.run_id || (m.run_ids && m.run_ids[0]) || '—';
  b.innerHTML = `<b>回放模式</b>：展示的是 ${esc(m.model || '模型')} 的一次真实运行录像（run ${esc(run)}），未调用模型。${m.recorded_at ? `录制于 ${esc(m.recorded_at)}。` : ''}配置 .env 中的模型 API Key 后即可实时生成。`;
  b.hidden = false;
}
function fillRecordedSelect(sel, list) {
  sel.innerHTML = '<option value="">选择一个已录制的场景…</option>' + list.map((x) => `<option value="${esc(JSON.stringify(x))}">${esc(topicText(x))}</option>`).join('');
}
function applyMode(m) {
  app.mode = m;
  const foot = $('footer-mode');
  if (m && m.mode === 'replay') {
    renderReplayBanner();
    const list = m.recorded_topics || [];
    fillRecordedSelect($('topic-select'), list);
    fillRecordedSelect($('followup-select'), list);
    $('topic').hidden = true; $('topic-select').hidden = false;
    $('followup-input').hidden = true; $('followup-select').hidden = false;
    $('desk-hint').textContent = '回放模式下只能查看已录制的场景。';
    document.querySelectorAll('input[name="genre"]').forEach((x) => { x.disabled = true; });
    $('inject').disabled = true;
    $('inject').parentNode.title = '回放模式下不可用：录像里是否含注入，以录像标注为准';
    foot.textContent = `回放模式 · 模型 ${m.model || '—'}`;
  } else {
    foot.textContent = m ? `${m.mock ? 'Mock 演示数据' : '实时模式'} · 模型 ${m.model || '—'}` : '后端未连接';
  }
  document.body.classList.toggle('is-replay', !!(m && m.mode === 'replay'));
}

function bindGlobalClicks() {
  document.addEventListener('click', (e) => {
    const cite = e.target.closest('.cite');
    if (cite) { const t = app.turns.get(cite.dataset.turn); if (t) openCite(t, cite.dataset.sid); return; }
    const pl = e.target.closest('.page-link');
    if (pl && pl.dataset.page) { openPage(Number(pl.dataset.page)); return; }
    const sg = e.target.closest('.sugg');
    if (sg && sg.dataset.msg) {
      if (isReplay()) return;
      if (sg.dataset.kind === 'clarify') { $('topic').value = sg.dataset.msg; deskSubmit(); }
      else send(sg.dataset.msg, {});
    }
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closePanel(); });
}

async function init() {
  $('desk').addEventListener('submit', deskSubmit);
  $('followup').addEventListener('submit', followupSubmit);
  $('new-session').addEventListener('click', newSession);
  $('panel-close').addEventListener('click', closePanel);
  bindGlobalClicks();
  try { applyMode(await fetchMode()); } catch (_) { applyMode(null); }
  try { renderSuggestions(await fetchSuggestions()); } catch (_) { renderSuggestions(FALLBACK_SUGGESTIONS); }
  // 自测 / 录屏用：?autorun=1 自动点「生成」（mock 与真实后端都可用）；&then=再出一版 FAQ|万悉科技的融资情况 依次追问
  if (QS.get('autorun') === '1' && !isReplay()) {
    await sleep(300);
    deskSubmit();
    while (app.busy) await sleep(100);
    for (const m of (QS.get('then') || '').split('|').filter(Boolean)) { await send(m, {}); }
    const cite = QS.get('cite'); // &cite=S4 自动打开第一轮该引用的页面面板
    if (cite) { const t = app.turns.values().next().value; if (t) await openCite(t, cite); }
    document.body.dataset.autorun = 'done';
  }
}

if (typeof document !== 'undefined') init();
