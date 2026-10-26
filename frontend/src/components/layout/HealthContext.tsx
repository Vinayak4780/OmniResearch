import { createContext, useContext, type ReactNode } from 'react';
import { usePolling } from '../../hooks/usePolling';
import { useResource } from '../../hooks/useResource';
import { api } from '../../services/api';
import type { HealthResponse } from '../../types/api';

interface HealthCtx {
  health: HealthResponse | undefined;
  error: unknown;
  reload: () => Promise<unknown>;
}

const Ctx = createContext<HealthCtx>({ health: undefined, error: null, reload: async () => undefined });

export function HealthProvider({ children }: { children: ReactNode }) {
  const r = useResource(() => api.health(), []);
  usePolling(() => r.reload(true), 30000, true);
  return <Ctx.Provider value={{ health: r.data, error: r.error, reload: () => r.reload(true) }}>{children}</Ctx.Provider>;
}

export function useHealth(): HealthCtx {
  return useContext(Ctx);
}
