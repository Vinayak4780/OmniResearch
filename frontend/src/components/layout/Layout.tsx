import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { useHealth } from './HealthContext';
import { HealthIndicator } from './HealthIndicator';

const NAV = [
  { to: '/', label: 'Notebooks', match: (p: string) => p === '/' || p.startsWith('/notebooks') },
  { to: '/settings', label: 'Settings', match: (p: string) => p.startsWith('/settings') },
];

function Banners() {
  const { health, error } = useHealth();
  // The notebooks home shows its own gentle prompt instead.
  const home = useLocation().pathname === '/';
  if (error) {
    return (
      <div className="banner banner-danger" role="alert">
        Cannot reach the OmniResearch API. Start the backend (<code>uvicorn app.main:app --port 8000</code>); this page re-checks every 30 seconds.
      </div>
    );
  }
  if (!health) return null;
  return (
    <>
      {!health.configured_tasks.synthesis && !home && (
        <div className="banner banner-warning" role="status">
          No AI model is connected yet, so research can't run. <Link to="/settings/connect">Connect an AI service</Link> — paste a key and
          it's set up in about a minute.
        </div>
      )}
      {health.status !== 'ok' && (
        <div className="banner banner-danger" role="status">
          System degraded:{!health.database && ' database unavailable.'}
          {!health.vector_store.ok && ` vector store (${health.vector_store.mode}) unavailable.`}
        </div>
      )}
    </>
  );
}

export function Layout() {
  const { pathname } = useLocation();
  const wide = pathname.startsWith('/notebooks/');
  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <Link to="/" className="brand" aria-label="OmniResearch home">
          <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="8" fill="var(--accent)" />
            <circle cx="14" cy="14" r="7" fill="none" stroke="white" strokeWidth="3" />
            <path d="M19 19l6 6" stroke="white" strokeWidth="3" strokeLinecap="round" />
          </svg>
          <span>OmniResearch</span>
        </Link>
        <nav className="mainnav" aria-label="Main">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === '/'} aria-current={n.match(pathname) ? 'page' : undefined} className={() => `navlink ${n.match(pathname) ? 'navlink-active' : ''}`}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <HealthIndicator />
      </header>
      <Banners />
      <main id="main" className={`main ${wide ? 'main-wide' : ''}`} tabIndex={-1}>
        <Outlet />
      </main>
    </div>
  );
}
