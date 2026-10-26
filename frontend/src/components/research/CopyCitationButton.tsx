import { useEffect, useState } from 'react';
import type { Evidence } from '../../types/api';
import { copyText, formatCitation } from '../../utils/citations';

export function CopyCitationButton({ evidence, className = 'btn btn-sm' }: { evidence: Evidence; className?: string }) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle');
  useEffect(() => {
    if (state === 'idle') return;
    const t = setTimeout(() => setState('idle'), 2000);
    return () => clearTimeout(t);
  }, [state]);
  return (
    <button
      type="button"
      className={className}
      title={formatCitation(evidence)}
      onClick={async () => setState((await copyText(formatCitation(evidence))) ? 'copied' : 'failed')}
    >
      {state === 'copied' ? 'Copied ✓' : state === 'failed' ? 'Copy failed' : 'Copy citation'}
      <span className="sr-only" role="status">
        {state === 'copied' ? `Citation ${evidence.id} copied` : ''}
      </span>
    </button>
  );
}
