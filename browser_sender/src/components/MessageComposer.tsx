import React, { useState, useRef } from 'react';
import { browserSession } from '../services/browserSession';
import {
  MessageType,
  ProgrammingLanguage,
  ALLOWED_LANGUAGES,
  MAX_MESSAGE_BYTES,
  createMessage,
  validateMessage,
} from '../types/browserMessage';

type SendStatus = 'idle' | 'sending' | 'sent' | 'error';

export const MessageComposer: React.FC = () => {
  const [msgType, setMsgType] = useState<MessageType>('question');
  const [language, setLanguage] = useState<ProgrammingLanguage>('python');
  const [content, setContent] = useState('');
  const [sendStatus, setSendStatus] = useState<SendStatus>('idle');
  const [error, setError] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const byteLength = new TextEncoder().encode(content).length;
  const tooLarge = byteLength > MAX_MESSAGE_BYTES;

  const handleSend = async () => {
    if (!content.trim()) {
      setError('Content cannot be empty.');
      return;
    }
    if (tooLarge) {
      setError(`Message is too large (${byteLength.toLocaleString()} / ${MAX_MESSAGE_BYTES.toLocaleString()} bytes).`);
      return;
    }

    const msg = createMessage(msgType, content, msgType === 'code' ? language : null);
    const errors = validateMessage(msg);
    if (errors.length > 0) {
      setError(errors.join(' '));
      return;
    }

    setError('');
    setSendStatus('sending');

    try {
      await browserSession.sendMessage(msg);
      setSendStatus('sent');
      // Clear only after successful send
      setContent('');
      setTimeout(() => setSendStatus('idle'), 2000);
    } catch (e: unknown) {
      const msg2 = e instanceof Error ? e.message : String(e);
      setError(msg2);
      setSendStatus('error');
    }
  };

  const handleClear = () => {
    setContent('');
    setError('');
    setSendStatus('idle');
    textareaRef.current?.focus();
  };

  return (
    <div className="message-composer">
      {/* Type selector */}
      <div className="type-selector" role="group" aria-label="Message type">
        {(['question', 'code', 'answer'] as MessageType[]).map((t) => (
          <button
            key={t}
            className={`type-btn ${msgType === t ? 'active' : ''}`}
            onClick={() => { setMsgType(t); setError(''); }}
          >
            {t === 'question' ? '❓ Question' : t === 'code' ? '💻 Code' : '💡 Answer'}
          </button>
        ))}
      </div>

      {/* Language selector for code */}
      {msgType === 'code' && (
        <div className="language-row">
          <label htmlFor="lang-select">Language:</label>
          <select
            id="lang-select"
            value={language}
            onChange={(e) => setLanguage(e.target.value as ProgrammingLanguage)}
          >
            {ALLOWED_LANGUAGES.map((l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
        </div>
      )}

      {/* Content editor */}
      <textarea
        ref={textareaRef}
        className={`content-editor ${msgType === 'code' ? 'code-editor' : ''}`}
        placeholder={
          msgType === 'code'
            ? 'Paste or type your code here…'
            : msgType === 'question'
            ? 'Type your question here…'
            : 'Type your answer here…'
        }
        value={content}
        onChange={(e) => { setContent(e.target.value); setError(''); }}
        rows={msgType === 'code' ? 12 : 5}
        spellCheck={msgType !== 'code'}
        aria-label="Message content"
      />

      {/* Size counter */}
      <div className={`byte-counter ${tooLarge ? 'over-limit' : ''}`}>
        {byteLength.toLocaleString()} / {MAX_MESSAGE_BYTES.toLocaleString()} bytes
        {tooLarge && ' ⚠ Too large'}
      </div>

      {error && <p className="error-msg" role="alert">{error}</p>}

      {/* Actions */}
      <div className="action-row">
        <button
          className="btn-primary"
          onClick={handleSend}
          disabled={sendStatus === 'sending' || !content.trim() || tooLarge}
          aria-busy={sendStatus === 'sending'}
        >
          {sendStatus === 'sending' ? 'Sending…' : sendStatus === 'sent' ? '✓ Sent!' : 'Send'}
        </button>
        <button
          className="btn-secondary"
          onClick={handleClear}
          disabled={sendStatus === 'sending'}
        >
          Clear
        </button>
      </div>
    </div>
  );
};
