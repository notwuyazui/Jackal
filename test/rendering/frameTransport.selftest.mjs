import assert from 'node:assert/strict';
import { acknowledgeFrame } from '../../render/render3D/src/frameTransport.ts';

const body = { view: '3d', tick: 0, version: 0, screenshot: 'first-frame' };
const ok = () => new Response(JSON.stringify({ ok: true }), { status: 200 });
const calls = [], delays = [];
await acknowledgeFrame(body, {
  fetchFn: async (url, options) => {
    calls.push({ url, options });
    if (calls.length <= 2) throw new TypeError('Failed to fetch');
    return ok();
  },
  pause: async delay => { delays.push(delay); },
});
assert.equal(calls.length, 3);
assert.deepEqual(delays, [150, 300]);
assert.ok(calls.every(call => call.url === '/api/ack' && call.options.body === JSON.stringify(body)));

let rejectedCalls = 0;
await assert.rejects(acknowledgeFrame(body, {
  fetchFn: async () => { rejectedCalls++; return new Response('stale session', { status: 403 }); },
}), /HTTP 403.*stale session/);
assert.equal(rejectedCalls, 1);

let proxyCalls = 0;
await acknowledgeFrame(body, {
  fetchFn: async () => ++proxyCalls === 1 ? new Response('proxy warming', { status: 502 }) : ok(),
  pause: async () => {},
});
assert.equal(proxyCalls, 2);

let failedCalls = 0;
await assert.rejects(acknowledgeFrame(body, {
  fetchFn: async () => { failedCalls++; throw new TypeError('Failed to fetch'); },
  pause: async () => {},
}), /\/api\/ack.*4.*Failed to fetch/);
assert.equal(failedCalls, 4);

const controller = new AbortController();
let abortedCalls = 0;
await assert.rejects(acknowledgeFrame(body, {
  signal: controller.signal,
  fetchFn: async () => { abortedCalls++; throw new TypeError('Failed to fetch'); },
  pause: async () => { controller.abort(new Error('window closed')); },
}), /window closed/);
assert.equal(abortedCalls, 1);

let timeoutCalls = 0;
await acknowledgeFrame(body, {
  timeoutMs: 10,
  fetchFn: async (_url, options) => {
    if (++timeoutCalls > 1) return ok();
    return new Promise((_resolve, reject) => options.signal.addEventListener('abort',
      () => reject(options.signal.reason), { once: true }));
  },
  pause: async () => {},
});
assert.equal(timeoutCalls, 2);
console.log('PASS: fetch/proxy recovery, identical retries, fatal errors, window closure and request timeout');
