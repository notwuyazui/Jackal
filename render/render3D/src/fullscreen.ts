/** Native launchers enter fullscreen first; a gesture handles manual URLs and
 * Safari sessions whose host has no macOS Accessibility permission. */
export function installFullscreenFallback() {
  const root = document.documentElement as HTMLElement & {
    webkitRequestFullscreen?: () => void | Promise<void>;
  };
  const owner = document as Document & { webkitFullscreenElement?: Element };
  const request = (event: Event) => {
    window.removeEventListener('pointerdown', request);
    window.removeEventListener('keydown', request);
    if (event instanceof KeyboardEvent && ['Escape', 'F11'].includes(event.key)) return;
    if (owner.fullscreenElement || owner.webkitFullscreenElement ||
        (window.outerHeight >= screen.height && window.outerWidth >= screen.width)) return;
    const enter = root.requestFullscreen || root.webkitRequestFullscreen;
    if (enter) {
      try {
        // Call synchronously inside the gesture, as required by Safari/Chromium.
        void Promise.resolve(enter.call(root)).catch(() => {});
      } catch { /* Fullscreen denial must not interrupt rendering. */ }
    }
  };
  window.addEventListener('pointerdown', request);
  window.addEventListener('keydown', request);
}
