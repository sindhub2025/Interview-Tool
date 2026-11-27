import React, { useState } from 'react';
import { browserSession, SessionState } from '../services/browserSession';

interface PairingFormProps {
  onConnected: () => void;
}

export const PairingForm: React.FC<PairingFormProps> = ({ onConnected }) => {
  const [code, setCode] = useState('');
  const [state, setState] = useState<SessionState>('idle');
  const [error, setError] = useState('');

  const handleConnect = async () => {
    const trimmed = code.trim().toUpperCase();
    if (!trimmed || trimmed.length !== 10) {
      setError('Enter the 10-character pairing code from GhostMic.');
      return;
    }
    setError('');
    setState('pairing');

    const unsub = browserSession.onStateChange((s, detail) => {
      setState(s);
      if (s === 'error') setError(detail);
      if (s === 'connected') {
        unsub();
        onConnected();
      }
    });

    try {
      await browserSession.join(trimmed);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  };

  const busy = state === 'authenticating' || state === 'pairing' || state === 'connecting';

  return (
    <div className="pairing-form">
      <h2>Connect to GhostMic</h2>
      <p className="hint">
        Open GhostMic on the other laptop, start a session, and enter the
        pairing code shown there.
      </p>
      <div className="input-row">
        <input
          type="text"
          className="pairing-input"
          placeholder="XXXXXXXXXX"
          maxLength={10}
          value={code}
          onChange={(e) => setCode(e.target.value.toUpperCase())}
          onKeyDown={(e) => e.key === 'Enter' && !busy && handleConnect()}
          disabled={busy}
          autoFocus
          aria-label="Pairing code"
        />
        <button
          className="btn-primary"
          onClick={handleConnect}
          disabled={busy || !code.trim()}
        >
          {busy ? 'Connecting…' : 'Connect'}
        </button>
      </div>
      {error && <p className="error-msg">{error}</p>}
      {busy && !error && (
        <p className="status-hint">{stateLabel(state)}</p>
      )}
    </div>
  );
};

function stateLabel(s: SessionState): string {
  switch (s) {
    case 'authenticating': return 'Authenticating…';
    case 'pairing': return 'Validating code…';
    case 'connecting': return 'Joining channel…';
    default: return '';
  }
}
