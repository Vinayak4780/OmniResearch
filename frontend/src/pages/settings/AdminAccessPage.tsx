import { useState, type FormEvent } from 'react';
import { useHealth } from '../../components/layout/HealthContext';
import { Alert, Card, ErrorMessage } from '../../components/ui';
import { api } from '../../services/api';
import { getAdminToken, setAdminToken } from '../../services/storage';

export function AdminAccessPage() {
  const { health } = useHealth();
  const [token, setToken] = useState(() => getAdminToken() ?? '');
  const [saved, setSaved] = useState<string | null>(null);
  const [show, setShow] = useState(false);
  const [checkResult, setCheckResult] = useState<'ok' | null>(null);
  const [checkError, setCheckError] = useState<unknown>(null);
  const [checking, setChecking] = useState(false);

  const save = (e: FormEvent) => {
    e.preventDefault();
    const ok = setAdminToken(token.trim() || null);
    setSaved(ok ? (token.trim() ? 'Token saved in this browser.' : 'Token removed.') : 'Could not access browser storage; the token is not persisted.');
    setCheckResult(null);
    setCheckError(null);
  };

  const check = async () => {
    setChecking(true);
    setCheckResult(null);
    setCheckError(null);
    try {
      await api.providers();
      setCheckResult('ok');
    } catch (e) {
      setCheckError(e);
    } finally {
      setChecking(false);
    }
  };

  return (
    <Card title="Admin access" subtitle="Admin endpoints manage providers, credentials, routing, reindexing and settings.">
      {health && (
        <Alert tone={health.admin_token_configured ? 'info' : 'neutral'}>
          {health.admin_token_configured
            ? 'This server requires an admin token (OMNI_ADMIN_TOKEN). Enter it below; it is sent as an Authorization: Bearer header on every request.'
            : 'This server has no admin token configured, so admin actions are only accepted from the local machine. A token is not needed when you use the app on localhost.'}
        </Alert>
      )}
      <form onSubmit={save} className="admin-form">
        <div className="field">
          <label htmlFor="admin-token" className="field-label">
            Admin token
          </label>
          <div className="input-group">
            <input id="admin-token" type={show ? 'text' : 'password'} autoComplete="off" spellCheck={false} value={token} onChange={(e) => setToken(e.target.value)} />
            <button type="button" className="btn" onClick={() => setShow(!show)} aria-pressed={show}>
              {show ? 'Hide' : 'Show'}
            </button>
          </div>
          <div className="field-hint">Stored in this browser's localStorage only.</div>
        </div>
        <div className="form-actions">
          <button type="submit" className="btn btn-primary">
            Save token
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => {
              setToken('');
              setAdminToken(null);
              setSaved('Token removed.');
            }}
          >
            Remove
          </button>
          <button type="button" className="btn btn-ghost" onClick={check} disabled={checking}>
            {checking ? 'Checking…' : 'Check access'}
          </button>
        </div>
      </form>
      {saved && <p className="small" role="status">{saved}</p>}
      {checkResult === 'ok' && <Alert tone="success">Admin access works.</Alert>}
      <ErrorMessage error={checkError} title="Admin access denied" />
    </Card>
  );
}
