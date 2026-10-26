import { useEffect, useRef } from 'react';

/**
 * Calls `fn` every `intervalMs` while `active` is true. The next tick is scheduled only after the
 * previous call settles, so slow responses never pile up.
 */
export function usePolling(fn: () => unknown | Promise<unknown>, intervalMs: number, active: boolean): void {
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    if (!active) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        await fnRef.current();
      } catch {
        /* callers surface their own errors */
      }
      if (!stopped) timer = setTimeout(tick, intervalMs);
    };
    timer = setTimeout(tick, intervalMs);
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [active, intervalMs]);
}
