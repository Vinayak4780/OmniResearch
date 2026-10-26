import { NavLink, Outlet } from 'react-router-dom';
import { PageHeader } from '../../components/ui';

const MAIN = [
  { to: 'connect', label: 'Connect AI' },
  { to: 'local-models', label: 'Local models' },
  { to: 'index', label: 'Document search index' },
  { to: 'access', label: 'Admin access' },
];

const ADVANCED = [
  { to: 'health', label: 'Provider health' },
];

function Item({ to, label }: { to: string; label: string }) {
  return (
    <NavLink to={to} className={({ isActive }) => `subnav-link ${isActive ? 'subnav-link-active' : ''}`}>
      {label}
    </NavLink>
  );
}

export function SettingsLayout() {
  return (
    <div>
      <PageHeader title="Settings" description="Connect an AI service, add free models that run on this computer, and manage document search." />
      <div className="settings-grid">
        <nav className="subnav" aria-label="Settings sections">
          {MAIN.map((i) => (
            <Item key={i.to} {...i} />
          ))}
          <div className="subnav-heading" id="subnav-advanced">
            Advanced
            <span className="subnav-sub">diagnostics</span>
          </div>
          <div role="group" aria-labelledby="subnav-advanced" className="subnav-group">
            {ADVANCED.map((i) => (
              <Item key={i.to} {...i} />
            ))}
          </div>
        </nav>
        <div className="settings-content">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
