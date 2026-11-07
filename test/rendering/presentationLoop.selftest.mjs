import assert from 'node:assert/strict';
import { startPresentationLoop } from '../../render/render3D/src/presentationLoop.ts';

let now = 0, serial = 0, timer, cleared = 0;
const frames = new Map(), rendered = [];
const loop = startPresentationLoop(time => rendered.push(time), {
  requestFrame(callback) { const id = ++serial; frames.set(id, callback); return id; },
  cancelFrame(id) { frames.delete(id); },
  now: () => now,
  every(callback, period) { assert.equal(period, 250); timer = callback; return 99; },
  clearEvery(id) { assert.equal(id, 99); cleared++; },
});
function raf(time) {
  now = time;
  const [id, callback] = frames.entries().next().value;
  frames.delete(id); callback(time);
}
raf(16);
assert.deepEqual(rendered, [16]);
now = 200; timer();
assert.deepEqual(rendered, [16]);
// The browser now stops dispatching RAF while its window is covered.
now = 500; timer();
now = 750; timer();
assert.deepEqual(rendered, [16, 500, 750]);
assert.equal(loop.fallbackFrames, 2);
assert.equal(frames.size, 1, 'only one RAF remains scheduled');
raf(766);
assert.deepEqual(rendered, [16, 500, 750, 766]);
now = 800; timer();
assert.equal(rendered.length, 4, 'foreground RAF is not duplicated by the fallback');
loop.stop(); loop.stop();
assert.equal(cleared, 1);
assert.equal(frames.size, 0);
now = 2000; timer();
assert.equal(rendered.length, 4);
console.log('PASS: covered-window rendering, single RAF after foreground return, complete loop cleanup');
