import React from 'react';
import { SessionState } from '../services/browserSession';

interface ConnectionStatusProps {
  state: SessionState;
  detail?: string;
  onDisconnect: () => void;
}

const STATE_LABELS: Record<SessionState, string> = {
  idle: 'Not connected',
  authenticating: 'Authenticating…',
  pairing: 'Pairing…',
  connecting: 'Connecting…',
  connected: 'Connected ✓',
  error: 'Error',
  disconnected: 'Disconnected',
};

const STATE_CLASSES: Record<SessionState, string> = {
  idle: 'status-idle',
  authenticating: 'status-busy',
  pairing: 'status-busy',
  connecting: 'status-busy',
  connected: 'status-ok',
  error: 'status-error',
  disconnected: 'status-idle',
};

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  state,
  detail,
  onDisconnect,
}) => {
  return (
    <div className={`connection-status ${STATE_CLASSES[state]}`}>
      <span className="status-dot" aria-hidden="true" />
      <span className="status-text">
        {STATE_LABELS[state]}
        {state === 'error' && detail ? `: ${detail}` : ''}
      </span>
      {state === 'connected' && (
        <button className="btn-disconnect" onClick={onDisconnect}>
          Disconnect
        </button>
      )}
    </div>
  );
};
