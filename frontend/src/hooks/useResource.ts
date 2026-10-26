import { useCallback, useEffect, useRef, useState } from 'react';

export interface Resource<T> {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  /** Re-fetch. `silent` keeps current data visible and does not toggle `loading`. */
  reload: (silent?: boolean) => Promise<T | undefined>;
  setData: (v: T | undefined | ((prev: T | undefined) => T | undefined)) => void;
}

/** Load async data on mount / when deps change; ignores results from superseded calls. */
export function useResource<T>(fetcher: () => Promise<T>, deps: readonly unknown[] = []): Resource<T> {
  const [data, setData] = useState<T | undefined>(undefined);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const seq = useRef(0);
  const mounted = useRef(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const reload = useCallback(async (silent = false) => {
    const my = ++seq.current;
    if (!silent) setLoading(true);
    try {
      const v = await fetcherRef.current();
      if (mounted.current && my === seq.current) {
        setData(v);
        setError(null);
      }
      return v;
    } catch (e) {
      if (mounted.current && my === seq.current) setError(e);
      return undefined;
    } finally {
      if (mounted.current && my === seq.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void reload();
    return () => {
      mounted.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, loading, reload, setData };
}
