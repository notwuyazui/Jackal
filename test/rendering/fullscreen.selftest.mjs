import assert from 'node:assert/strict';
import { installFullscreenFallback } from '../../render/render3D/src/fullscreen.ts';

globalThis.KeyboardEvent = class extends Event {
  constructor(key) { super('keydown'); this.key = key; }
};

function scene({ webkit = false, native = false, active = false, reject = false } = {}) {
  const listeners = new Map();
  const calls = [];
  const root = {};
  root[webkit ? 'webkitRequestFullscreen' : 'requestFullscreen'] = function () {
    calls.push(this);
    return reject ? Promise.reject(new Error('denied')) : Promise.resolve();
  };
  globalThis.document = { documentElement: root, fullscreenElement: active ? root : null };
  globalThis.screen = { width: 1920, height: 1080 };
  globalThis.window = {
    outerWidth: native ? 1920 : 1280, outerHeight: native ? 1080 : 800,
    addEventListener(name, callback) { listeners.set(name, callback); },
    removeEventListener(name) { listeners.delete(name); },
  };
  installFullscreenFallback();
  assert.equal(calls.length, 0); // Browsers require a real gesture, never on load.
  return { listeners, calls, root };
}

for (const webkit of [false, true]) {
  const { listeners, calls, root } = scene({ webkit });
  listeners.get('pointerdown')(new Event('pointerdown'));
  assert.deepEqual(calls, [root]);
  assert.equal(listeners.size, 0);
}
for (const options of [{ native: true }, { active: true }]) {
  const { listeners, calls } = scene(options);
  listeners.get('pointerdown')(new Event('pointerdown'));
  assert.equal(calls.length, 0);
}
for (const key of ['Escape', 'F11']) {
  const { listeners, calls } = scene();
  listeners.get('keydown')(new KeyboardEvent(key));
  assert.equal(calls.length, 0);
  assert.equal(listeners.size, 0); // Never force fullscreen again after an exit.
}
const { listeners, calls } = scene({ reject: true });
listeners.get('keydown')(new KeyboardEvent('Enter'));
await new Promise(resolve => setTimeout(resolve, 0));
assert.equal(calls.length, 1); // Denial is handled without an unhandled rejection.
console.log('PASS: gesture fullscreen, Safari fallback, native fullscreen and exit handling');
