import type { Frame } from './protocol.ts';
import { commitInset } from './inset.ts';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
document.title = 'Claude of Tanks · M1A1 同步战场';
const inset = $<HTMLCanvasElement>('jackal-inset');
const insetContext = inset.getContext('2d', { alpha: false })!;
const decoder = new Image(); // Never replace an image currently on screen.
let insetCommits = 0;
const healthElements = new Map<number, { root: HTMLSpanElement; value: HTMLElement }>();
let healthRun = -1;
if (new URLSearchParams(window.location.search).get('inset') === '0') {
  $('subtitle').textContent = 'CoT 3D 画面 · 自动运行';
}

async function acknowledge(body: unknown) {
  const response = await fetch('/api/ack', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
}

let reportViewerFailure: ((error: unknown) => void) | null = null;
function fail(error: unknown) {
  $('loading').hidden = false;
  $('loading').textContent = String(error);
  $('connection').textContent = '同步异常 · 请检查错误';
  console.error(error);
}

window.addEventListener("error", event => (reportViewerFailure || fail)(event.error || event.message));
window.addEventListener("unhandledrejection", event => (reportViewerFailure || fail)(event.reason));

async function displayJackal(frame: Frame) {
  if (!frame.jackal_png.startsWith('data:image/png;base64,')) throw new Error('Jackal 2D 帧不是 PNG');
  const encoded = frame.jackal_png.slice('data:image/png;base64,'.length);
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  const sha256 = [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
  if (sha256 !== frame.jackal_png_sha256) throw new Error(`Jackal 2D 帧校验失败：STEP ${frame.tick}`);
  decoder.src = frame.jackal_png;
  await decoder.decode();
  const size = [decoder.naturalWidth, decoder.naturalHeight];
  if (size[0] !== frame.manifest.map.size[0] || size[1] !== frame.manifest.map.size[1]) {
    throw new Error(`Jackal 2D 画面尺寸错误：${size.join('×')}`);
  }
  return { mode: 'original-pygame-frame', tick: frame.tick, png_sha256: sha256, size,
    presentation: 'atomic-canvas-copy', commits: ++insetCommits };
}

function commitJackal(frame: Frame) {
  // Keep the prior complete frame until the new PNG AND 3D scene are ready.
  // Only map changes resize the backing canvas, within the same display task.
  commitInset(inset, insetContext, decoder, frame.tick);
}

function hud(frame: Frame) {
  const { result } = frame;
  const showInset = frame.manifest.show_inset !== false;
  $('subtitle').textContent = showInset ? 'Jackal 原版 2D 画面内嵌 · 自动运行' : 'CoT 3D 画面 · 自动运行';
  if (!showInset) inset.hidden = true;
  $('clock').textContent = `${frame.time.toFixed(2)} s · STEP ${frame.tick}`;
  $('score').textContent = `蓝方 ${result.alive.PLAYER} : ${result.alive.ENEMY} 红方`;
  $('digest').textContent = `STATE ${frame.hash} · Δt ${frame.manifest.dt}s`;
  $('connection').textContent = result.finished ? '对战结束 · 已同步' : frame.tick === 0 ? '同步就绪 · 自动开始' : showInset ? 'Jackal 2D / CoT 3D 同步' : 'CoT 3D · 物理状态已同步';
  const allies = frame.units.filter(u => u.team === 'PLAYER').length;
  const enemies = frame.units.filter(u => u.team === 'ENEMY').length;
  $('scenario').textContent = `${frame.manifest.scene.name} · ${allies} 对 ${enemies} · ${frame.manifest.controller || 'EnemyAI Lv.5'}`;
  if (healthRun !== frame.run_id) {
    healthElements.clear(); $('healths').replaceChildren(); healthRun = frame.run_id;
  }
  for (const unit of frame.units) {
    let entry = healthElements.get(unit.id);
    if (!entry) {
      const root = document.createElement('span'), value = document.createElement('b');
      root.append(document.createTextNode(`#${unit.id}`), value);
      $('healths').appendChild(root);
      entry = { root, value }; healthElements.set(unit.id, entry);
    }
    entry.root.className = unit.team === 'ENEMY' ? 'enemy' : '';
    entry.value.textContent = `${unit.alive ? unit.health : 0} HP`;
  }
  const followed = frame.units.find(u => u.id === frame.manifest.follow_id)!;
  $('focus').textContent = `炮塔跟随 · #${followed.id} · ${followed.alive ? `${followed.health} HP` : '已被击毁'} · 装填 ${followed.cooldown.toFixed(2)} s`;
  $('result').hidden = !result.finished;
  if (result.finished) $('result').innerHTML = `<strong>${result.winner === 'ENEMY' ? '红方获胜' : result.winner === 'PLAYER' ? '蓝方获胜' : '战斗平局'}</strong><small>${frame.time.toFixed(2)} 秒 · ${frame.tick} 步 · 存活 ${result.alive.PLAYER} : ${result.alive.ENEMY}</small>`;
}

async function boot() {
  const viewToken = new URLSearchParams(window.location.search).get('token') || '';
  const windowClient = crypto.randomUUID();
  function windowEvent(action: 'open' | 'close') {
    const body = JSON.stringify({ view_token: viewToken, client_id: windowClient, action });
    if (action === 'close' && navigator.sendBeacon('/api/window',
      new Blob([body], { type: 'application/json' }))) return;
    void fetch('/api/window', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body, keepalive: true }).catch(() => {});
  }
  windowEvent('open');
  const events = new EventSource(`/api/events?view=3d&token=${encodeURIComponent(viewToken)}`);
  let view: Awaited<ReturnType<typeof import('./view3d.ts')['createView']>> | null = null;
  let pending: Frame | null = null;
  let processing = false;
  let stopped = false;
  let terminalError: unknown = null;
  let receivedVersion = -1;
  let lastStreamFrameAt = 0;
  let retryTimer = 0;
  const requests = new AbortController();
  let appliedCursor = 0, appliedRun = -1;

  function stop() {
    if (stopped) return;
    stopped = true;
    clearTimeout(retryTimer);
    requests.abort();
    events.close();
    view?.dispose();
  }
  events.addEventListener('closed', event => {
    stop();
    const status = JSON.parse((event as MessageEvent<string>).data);
    if (status.error) { fail(status.error); return; }
    if ($('result').hidden) $('connection').textContent = '战斗服务已停止';
  });

  function reportFailure(error: unknown) {
    if (terminalError !== null || stopped) return;
    terminalError = error;
    // Scene/model/WebGL failures must wake Python's lockstep wait immediately.
    // This request intentionally survives aborting the normal frame requests.
    void fetch('/api/error', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ view_token: viewToken, message: String(error).slice(0, 4000) || '未知渲染错误' }),
      keepalive: true }).catch(() => {});
    stop();
    fail(error);
  }
  reportViewerFailure = reportFailure;

  function receive(frame: Frame) {
    // The first frame may arrive from HTTP and SSE. Never apply it twice,
    // replay events, or let an older HTTP response replace a newer SSE pose.
    if (stopped || frame.version <= receivedVersion) return;
    receivedVersion = frame.version;
    pending = frame;
    if (!view) $('connection').textContent = '战斗首帧已读取 · 正在加载场景';
    void drain();
  }

  async function drain() {
    if (processing || !view || stopped) return;
    processing = true;
    try {
      while (pending) {
        const frame: Frame = pending;
        pending = null;
        if (frame.run_id !== appliedRun) { appliedCursor = 0; appliedRun = frame.run_id; }
        $("loading").textContent = "正在同步战斗画面…";
        const jackal_inset = frame.manifest.show_inset === false ? undefined : await displayJackal(frame);
        $("loading").textContent = "正在绘制战场…";
        const queued = frame.pending_events?.filter(e => e.seq > appliedCursor);
        const visualFrame = queued ? { ...frame, events: queued } : frame;
        const receipt = await view.draw(visualFrame, () => {
          if (jackal_inset) commitJackal(frame);
          hud(frame);
          $('loading').hidden = true;
        });
        if (stopped) return;
        await acknowledge({ view: '3d', version: frame.version, tick: frame.tick, hash: frame.hash,
          event_cursor: frame.event_cursor, jackal_inset, ...receipt });
        appliedCursor = frame.event_cursor;
      }
    } catch (error) {
      if (!stopped) reportFailure(error);
    } finally {
      processing = false;
    }
  }
  events.onmessage = event => {
    // One frame may be decoded and one may wait; events are retained by the server until ACK.
    lastStreamFrameAt = Date.now();
    try { receive(JSON.parse(event.data)); }
    catch (error) { reportFailure(error); }
  };
  events.onopen = () => {
    if (receivedVersion < 0) $('connection').textContent = '战斗服务已连接 · 正在加载场景';
  };
  events.onerror = () => {
    if (stopped) return;
    lastStreamFrameAt = 0;
    if ($('result').hidden) $('connection').textContent = '正在恢复战斗连接…';
  };

  async function pollFrame() {
    if (stopped || !$('result').hidden) return;
    let advanced = false;
    try {
      // Always fetch the initial frame. Later, use HTTP only if pushed frames
      // have stalled. The server waits for a newer version before responding.
      if (receivedVersion < 0 || Date.now() - lastStreamFrameAt > 1500) {
        const response = await fetch(`/api/frame?token=${encodeURIComponent(viewToken)}&after=${receivedVersion}`,
          { cache: 'no-store', signal: requests.signal });
        if (response.status === 403) {
          reportFailure('战斗会话已失效，请重新运行测试命令以打开新的战斗窗口。');
          return;
        }
        if (response.status === 503) {
          reportFailure((await response.json()).error || '3D 战斗服务已失败');
          return;
        }
        if (response.status === 200) {
          const frame: Frame = await response.json();
          advanced = frame.version > receivedVersion;
          receive(frame);
        }
        else if (response.status !== 202) throw new Error(`战斗服务 HTTP ${response.status}`);
      }
    } catch (error) {
      if (!stopped && $('result').hidden) {
        $('connection').textContent = '本地战斗服务连接中断 · 自动重连';
        if (view && receivedVersion < 0) {
          $('loading').textContent = '暂时无法读取战斗首帧，请查看运行命令的终端。正在自动重连…';
        }
      }
    } finally {
      if (!stopped && $('result').hidden) {
        retryTimer = window.setTimeout(() => void pollFrame(),
          Date.now() - lastStreamFrameAt > 1500 ? advanced ? 0 : 100 : 500);
      }
    }
  }
  window.addEventListener('pagehide', () => { windowEvent('close'); stop(); }, { once: true });
  void pollFrame();
  try {
    const { createView } = await import('./view3d.ts');
    if (stopped) return;
    view = await createView($('viewport'));
    if (stopped) {
      view.dispose();
      if (terminalError !== null) fail(terminalError);
      return;
    }
    await drain();
  } catch (error) {
    reportFailure(error);
  }
}

boot().catch(fail);
