import { Suspense, lazy, type ComponentType, type LazyExoticComponent } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { HealthProvider } from './components/layout/HealthContext';
import { Layout } from './components/layout/Layout';
import { Spinner } from './components/ui';
import { DocumentViewerPage } from './pages/DocumentViewerPage';
import { NotFoundPage } from './pages/NotFoundPage';
import { NotebookPage } from './pages/NotebookPage';
import { NotebooksPage } from './pages/NotebooksPage';
import { ResearchRedirect } from './pages/ResearchRedirect';

/** Code-split a named export (created once at module load, never during render). */
function named<K extends string>(load: () => Promise<Record<K, ComponentType>>, name: K): LazyExoticComponent<ComponentType> {
  return lazy(() => load().then((m) => ({ default: m[name] })));
}

const SettingsLayout = named(() => import('./pages/settings/SettingsLayout'), 'SettingsLayout');
const ConnectPage = named(() => import('./pages/settings/ConnectPage'), 'ConnectPage');
const LocalModelsPage = named(() => import('./pages/settings/LocalModelsPage'), 'LocalModelsPage');

const HealthPage = named(() => import('./pages/settings/HealthPage'), 'HealthPage');
const IndexPage = named(() => import('./pages/settings/IndexPage'), 'IndexPage');
const AdminAccessPage = named(() => import('./pages/settings/AdminAccessPage'), 'AdminAccessPage');

function L({ C }: { C: LazyExoticComponent<ComponentType> }) {
  return (
    <Suspense fallback={<Spinner />}>
      <C />
    </Suspense>
  );
}

export function App() {
  return (
    <HealthProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<NotebooksPage />} />
          <Route path="notebooks/:id" element={<NotebookPage />} />
          <Route path="documents" element={<Navigate to="/" replace />} />
          <Route path="documents/:id" element={<Navigate to="/" replace />} />
          <Route path="documents/:id/view" element={<DocumentViewerPage />} />
          <Route path="research" element={<Navigate to="/" replace />} />
          <Route path="research/:jobId" element={<ResearchRedirect />} />
          <Route path="settings" element={<L C={SettingsLayout} />}>
            <Route index element={<Navigate to="connect" replace />} />
            <Route path="connect" element={<L C={ConnectPage} />} />
            <Route path="local-models" element={<L C={LocalModelsPage} />} />
            <Route path="providers" element={<Navigate to="/settings/connect" replace />} />
            <Route path="routing" element={<Navigate to="/settings/connect" replace />} />
            <Route path="health" element={<L C={HealthPage} />} />
            <Route path="ocr" element={<Navigate to="/settings/connect" replace />} />
            <Route path="index" element={<L C={IndexPage} />} />
            <Route path="web" element={<Navigate to="/settings/connect" replace />} />
            <Route path="access" element={<L C={AdminAccessPage} />} />
          </Route>
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </HealthProvider>
  );
}
