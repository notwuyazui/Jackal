/** Keep lockstep presentation alive when a covered browser stops delivering RAF. */
interface LoopPorts {
  requestFrame: (callback: FrameRequestCallback) => number;
  cancelFrame: (id: number) => void;
  now: () => number;
  every: (callback: () => void, milliseconds: number) => number;
  clearEvery: (id: number) => void;
}

export function startPresentationLoop(render: (time: number) => void, ports: Partial<LoopPorts> = {}) {
  const request = ports.requestFrame ?? (callback => window.requestAnimationFrame(callback));
  const cancel = ports.cancelFrame ?? (id => window.cancelAnimationFrame(id));
  const now = ports.now ?? (() => performance.now());
  const every = ports.every ?? ((callback, milliseconds) => window.setInterval(callback, milliseconds));
  const clearEvery = ports.clearEvery ?? (id => window.clearInterval(id));
  let stopped = false, lastFrame = now(), fallbackFrames = 0;
  function tick(time: number) {
    if (stopped) return;
    lastFrame = time;
    render(time);
    if (!stopped) frame = request(tick);
  }
  let frame = request(tick);
  const timer = every(() => {
    const time = now();
    if (stopped || time - lastFrame < 250) return;
    // Cancel the dormant RAF before replacing it; returning to the foreground
    // must never create a second render loop or present the same frame twice.
    cancel(frame);
    fallbackFrames++;
    tick(time);
  }, 250);
  return {
    get fallbackFrames() { return fallbackFrames; },
    stop() {
      if (stopped) return;
      stopped = true;
      cancel(frame);
      clearEvery(timer);
    },
  };
}
