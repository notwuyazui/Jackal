/** Retry an identical, idempotent render receipt without drawing its frame twice. */
interface ReceiptOptions {
  signal?: AbortSignal;
  fetchFn?: typeof fetch;
  pause?: (milliseconds: number, signal?: AbortSignal) => Promise<void>;
  timeoutMs?: number;
  attempts?: number;
}

class ReceiptRejected extends Error {}

function retryDelay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(signal.reason); return; }
    const abort = () => { clearTimeout(timer); reject(signal?.reason); };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', abort);
      resolve();
    }, milliseconds);
    signal?.addEventListener('abort', abort, { once: true });
  });
}

export async function acknowledgeFrame(body: unknown, {
  signal, fetchFn = fetch, pause = retryDelay, timeoutMs = 7000, attempts = 4,
}: ReceiptOptions = {}): Promise<void> {
  const encoded = JSON.stringify(body);
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt++) {
    if (signal?.aborted) throw signal.reason;
    const request = new AbortController();
    const abort = () => request.abort(signal?.reason);
    signal?.addEventListener('abort', abort, { once: true });
    const timeout = setTimeout(() => request.abort(new Error('呈现回执请求超时')), timeoutMs);
    try {
      const response = await fetchFn('/api/ack', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: encoded, cache: 'no-store', signal: request.signal,
      });
      if (!response.ok) {
        const message = await response.text();
        const detail = `呈现回执 HTTP ${response.status}: ${message.slice(0, 300)}`;
        // Invalid sessions/hashes are real protocol errors; retrying cannot fix them.
        if (response.status < 500 && response.status !== 408 && response.status !== 429) {
          throw new ReceiptRejected(detail);
        }
        throw new Error(detail);
      }
      await response.json();
      return;
    } catch (error) {
      if (signal?.aborted) throw signal.reason;
      if (error instanceof ReceiptRejected) throw error;
      lastError = error;
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener('abort', abort);
    }
    if (attempt + 1 < attempts) await pause(150 * 2 ** attempt, signal);
  }
  throw new Error(`本地 3D 呈现回执传输失败（/api/ack，尝试 ${attempts} 次）：${String(lastError)}`,
    { cause: lastError });
}
