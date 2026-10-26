import { Link } from 'react-router-dom';
import { EmptyState } from '../components/ui';

export function NotFoundPage() {
  return (
    <EmptyState title="Page not found">
      <Link to="/">Go to your notebooks</Link>
    </EmptyState>
  );
}
