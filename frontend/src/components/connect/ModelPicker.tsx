import { useId, useState } from 'react';
import type { DiscoveredModel } from '../../types/quickconnect';
import { roleOptions, type RoleDef } from './connectLogic';

const FILTER_THRESHOLD = 10;

interface Props {
  def: RoleDef;
  models: DiscoveredModel[];
  recommended: string | null | undefined;
  value: string | null;
  onChange: (v: string | null) => void;
  disabled?: boolean;
  /** Ids whose test is still running (shown as "testing…"). */
  testing?: ReadonlySet<string>;
}

/** A role's model dropdown: usable models first, unavailable ones disabled, with a filter box for long lists. */
export function ModelPicker({ def, models, recommended, value, onChange, disabled, testing }: Props) {
  const id = useId();
  const [query, setQuery] = useState('');
  const all = roleOptions(models, def, recommended, '', value, testing);
  const opts = roleOptions(models, def, recommended, query, value, testing);
  const tested = opts.filter((o) => o.state === 'tested');
  const untested = opts.filter((o) => o.state === 'untested');
  const unavailable = opts.filter((o) => o.state === 'unavailable');
  const selected = all.find((o) => o.id === value);
  const noChoices = all.filter((o) => !o.disabled).length === 0 && !def.specials?.length;

  return (
    <div className="role-row" data-role={def.role}>
      <div className="role-text">
        <label htmlFor={`${id}-select`} className="role-label">
          {def.label}
          {!def.noneLabel && !def.specials && <span className="required" aria-hidden="true"> *</span>}
        </label>
        <div className="small muted" id={`${id}-hint`}>
          {def.hint}
        </div>
      </div>
      <div className="role-control">
        {all.length > FILTER_THRESHOLD && (
          <input
            type="search"
            className="role-filter"
            placeholder={`Filter ${all.length} models…`}
            aria-label={`Filter models for “${def.label}”`}
            value={query}
            disabled={disabled}
            onChange={(e) => setQuery(e.target.value)}
          />
        )}
        <select
          id={`${id}-select`}
          value={value ?? ''}
          disabled={disabled}
          aria-describedby={`${id}-hint`}
          onChange={(e) => onChange(e.target.value || null)}
        >
          {def.specials ? (
            def.specials.map((x) => (
              <option key={x.value} value={x.value}>
                {x.label}
              </option>
            ))
          ) : def.noneLabel ? (
            <option value="">{def.noneLabel}</option>
          ) : (
            !value && <option value="">Choose a model…</option>
          )}
          {tested.length > 0 && (
            <optgroup label="Tested and working">
              {tested.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </optgroup>
          )}
          {untested.length > 0 && (
            <optgroup label="Not tested (may work)">
              {untested.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.label}
                </option>
              ))}
            </optgroup>
          )}
          {unavailable.length > 0 && (
            <optgroup label="Not available with this key">
              {unavailable.map((o) => (
                <option key={o.id} value={o.id} disabled>
                  {o.label}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        {query && opts.length === 0 && <div className="small muted">No models match “{query}”.</div>}
        {noChoices && !def.noneLabel && <div className="small text-danger">None of this service's models can do this with your key.</div>}
        {selected?.state === 'untested' && (
          <div className="small text-warning">
            {testing?.has(selected.id) ? 'Being tested right now…' : 'Not tested yet. It is listed by the service but may not be enabled for your key.'}
          </div>
        )}
        {selected?.note?.startsWith('answers') && (
          <div className="small text-warning">This model {selected.note}; reports may fail. Prefer a ✓ model without a note.</div>
        )}
        {selected?.state === 'tested' && selected.note && !selected.note.startsWith('answers') && def.capability !== 'embedding' && (
          <div className="small muted">Test note: {selected.note}</div>
        )}
      </div>
    </div>
  );
}
