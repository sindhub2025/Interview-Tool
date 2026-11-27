import React, { useState } from 'react';
import { PairingForm } from './components/PairingForm';
import { MessageComposer } from './components/MessageComposer';
import { ConnectionStatus } from './components/ConnectionStatus';
import { browserSession, SessionState } from './services/browserSession';
import './styles.css';

export const App: React.FC = () => {
  const [connected, setConnected] = useState(false);
  const [sessionState, setSessionState] = useState<SessionState>('idle');
  const [stateDetail, setStateDetail] = useState('');

  React.useEffect(() => {
    const unsub = browserSession.onStateChange((s, detail) => {
      setSessionState(s);
      setStateDetail(detail);
      if (s === 'disconnected' || s === 'error') {
        setConnected(false);
      }
    });
    return unsub;
  }, []);

  const handleDisconnect = async () => {
    await browserSession.disconnect();
    setConnected(false);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>👻 GhostMic Remote</h1>
        <p className="subtitle">Send questions, code and answers to GhostMic</p>
      </header>

      <main className="app-main">
        <ConnectionStatus
          state={sessionState}
          detail={stateDetail}
          onDisconnect={handleDisconnect}
        />

        {!connected ? (
          <PairingForm onConnected={() => setConnected(true)} />
        ) : (
          <MessageComposer />
        )}
      </main>

      <footer className="app-footer">
        <p>
          Messages are transmitted through Supabase Realtime and are{' '}
          <strong>not permanently stored</strong>. They disappear when the GhostMic
          session ends.
        </p>
      </footer>
    </div>
  );
};
