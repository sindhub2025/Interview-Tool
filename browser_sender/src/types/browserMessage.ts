// Message contract — shared between sender and receiver.
// IMPORTANT: Keep this file in sync with ghostmic/domain/browser_message.py

export const PROTOCOL_VERSION = 1;
export const MAX_MESSAGE_BYTES = 102_400; // 100 KB

export type MessageType = 'question' | 'code' | 'answer';

export const ALLOWED_LANGUAGES = [
  'python', 'javascript', 'typescript', 'java', 'c', 'cpp', 'csharp',
  'go', 'rust', 'sql', 'bash', 'powershell', 'json', 'yaml',
  'html', 'css', 'php', 'ruby', 'plaintext',
] as const;

export type ProgrammingLanguage = typeof ALLOWED_LANGUAGES[number];

export interface BrowserMessage {
  protocol_version: number;
  message_id: string;
  type: MessageType;
  language: ProgrammingLanguage | null;
  content: string;
  created_at: string;
}

export function createMessage(
  type: MessageType,
  content: string,
  language: ProgrammingLanguage | null = null
): BrowserMessage {
  return {
    protocol_version: PROTOCOL_VERSION,
    message_id: crypto.randomUUID(),
    type,
    language: type === 'code' ? language : null,
    content,
    created_at: new Date().toISOString(),
  };
}

export function validateMessage(msg: BrowserMessage): string[] {
  const errors: string[] = [];

  if (msg.protocol_version !== PROTOCOL_VERSION) {
    errors.push(`protocol_version must be ${PROTOCOL_VERSION}.`);
  }
  if (!msg.message_id || !/^[0-9a-f-]{36}$/i.test(msg.message_id)) {
    errors.push('message_id must be a valid UUID.');
  }
  if (!['question', 'code', 'answer'].includes(msg.type)) {
    errors.push('type must be question, code or answer.');
  }
  if (!msg.content || msg.content.trim() === '') {
    errors.push('content must not be empty.');
  }
  const byteLength = new TextEncoder().encode(msg.content).length;
  if (byteLength > MAX_MESSAGE_BYTES) {
    errors.push(
      `content exceeds maximum size of ${MAX_MESSAGE_BYTES} bytes (got ${byteLength}).`
    );
  }
  if (msg.type === 'code' && msg.language !== null) {
    if (!ALLOWED_LANGUAGES.includes(msg.language as ProgrammingLanguage)) {
      errors.push(`language '${msg.language}' is not supported.`);
    }
  }
  return errors;
}
