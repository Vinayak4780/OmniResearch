import { useEffect, useState } from 'react';
import { Navigate, useParams } from 'react-router-dom';
import { Spinner } from '../components/ui';
import { api } from '../services/api';

/** Old /research/:jobId links: open the report inside its notebook, or go home. */
export function ResearchRedirect() {
  const { jobId } = useParams();
  const [to, setTo] = useState<string | null>(jobId ? null : '/');
  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    api
      .researchJob(jobId)
      .then((j) => {
        const nb = j.options?.notebook_id;
        if (alive) setTo(nb ? `/notebooks/${encodeURIComponent(nb)}?report=${encodeURIComponent(jobId)}` : '/');
      })
      .catch(() => alive && setTo('/'));
    return () => {
      alive = false;
    };
  }, [jobId]);
  return to ? <Navigate to={to} replace /> : <Spinner label="Opening report…" />;
}
