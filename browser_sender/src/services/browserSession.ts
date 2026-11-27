// browserSession.ts — session lifecycle and Realtime channel management.
//
// Design:
// - Authenticate anonymously.
// - Invoke join-browser-session Edge Function.
// - Subscribe to the private Realtime channel.
// - Broadcast browser_message events.
// - Never store message content.

import { RealtimeChannel } from '@supabase/supabase-js';
import { supabase } from './supabaseClient';
import { BrowserMessage } from '../types/browserMessage';

const BROWSER_MESSAGE_EVENT = 'browser_message';
const SENDER_JOINED_EVENT = 'sender_joined';

export type SessionState =
  | 'idle'
  | 'authenticating'
  | 'pairing'
  | 'connecting'
  | 'connected'
  | 'error'
  | 'disconnected';

export interface SessionResult {
  session_id: string;
  channel_topic: string;
  expires_at: string;
}

class BrowserSession {
  private channel: RealtimeChannel | null = null;
  private channelTopic: string | null = null;
  private _state: SessionState = 'idle';
  private stateListeners: Array<(s: SessionState, detail: string) => void> = [];

  get state(): SessionState {
    return this._state;
  }

  onStateChange(cb: (s: SessionState, detail: string) => void): () => void {
    this.stateListeners.push(cb);
    return () => {
      this.stateListeners = this.stateListeners.filter((l) => l !== cb);
    };
  }

  private setState(s: SessionState, detail = '') {
    this._state = s;
    this.stateListeners.forEach((l) => l(s, detail));
  }

  async join(pairingCode: string): Promise<void> {
    this.setState('authenticating', 'Signing in anonymously…');

    const { error: authError } = await supabase.auth.signInAnonymously();
    if (authError) {
      this.setState('error', `Auth failed: ${authError.message}`);
      throw authError;
    }

    this.setState('pairing', 'Validating pairing code…');

    const { data: sessionData, error: fnError } =
      await supabase.functions.invoke<SessionResult>('join-browser-session', {
        body: { pairing_code: pairingCode },
      });

    if (fnError || !sessionData) {
      const msg = fnError?.message || 'Invalid or expired pairing code.';
      this.setState('error', msg);
      throw new Error(msg);
    }

    this.channelTopic = sessionData.channel_topic;
    this.setState('connecting', 'Joining channel…');

    await this._subscribeChannel();
  }

  private async _subscribeChannel(): Promise<void> {
    if (!this.channelTopic) return;

    return new Promise((resolve, reject) => {
      const ch = supabase.channel(this.channelTopic!, {
        config: { private: true },
      });

      ch.subscribe((status) => {
        if (status === 'SUBSCRIBED') {
          this.channel = ch;
          // Announce presence to the receiver
          ch.send({
            type: 'broadcast',
            event: SENDER_JOINED_EVENT,
            payload: {},
          }).catch(() => {});
          this.setState('connected', 'Ready to send messages.');
          resolve();
        } else if (status === 'CHANNEL_ERROR' || status === 'TIMED_OUT') {
          this.setState('error', `Channel error: ${status}`);
          reject(new Error(`Channel subscription failed: ${status}`));
        }
      });
    });
  }

  async sendMessage(msg: BrowserMessage): Promise<void> {
    if (!this.channel || this._state !== 'connected') {
      throw new Error('Not connected. Please pair first.');
    }

    const result = await this.channel.send({
      type: 'broadcast',
      event: BROWSER_MESSAGE_EVENT,
      payload: msg,
    });

    if (result !== 'ok') {
      throw new Error(`Send failed: ${result}`);
    }
  }

  async disconnect(): Promise<void> {
    this.setState('disconnected', 'Disconnected.');
    if (this.channel) {
      await this.channel.unsubscribe();
      this.channel = null;
    }
    await supabase.auth.signOut();
    this.channelTopic = null;
  }
}

// Singleton for the lifetime of the page
export const browserSession = new BrowserSession();
