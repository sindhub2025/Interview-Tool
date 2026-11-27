// supabase/functions/create-browser-session/index.ts
//
// Edge Function: create-browser-session
//
// Called by the GhostMic desktop client to start a new browser session.
//
// Steps:
//  1. Validate the caller JWT (must be an authenticated Supabase user).
//  2. Generate a cryptographically secure pairing code.
//  3. Hash it with SHA-256 before storage (never store plaintext).
//  4. Insert a session row for the authenticated user.
//  5. Return the session_id, plaintext pairing_code, channel_topic and expires_at.
//
// The plaintext pairing_code is returned only in this response and is NEVER
// written to the database or any log.

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { crypto } from "https://deno.land/std@0.177.0/crypto/mod.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

// Crockford Base32 alphabet (lowercase)
const CROCKFORD = "0123456789abcdefghjkmnpqrstvwxyz";
const CODE_LENGTH = 10;
const DEFAULT_TTL_MINUTES = 10;
const MAX_TTL_MINUTES = 30;

function generatePairingCode(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(CODE_LENGTH));
  return Array.from(bytes)
    .map((b) => CROCKFORD[b % CROCKFORD.length])
    .join("")
    .toUpperCase();
}

async function sha256hex(text: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(text);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "authorization, content-type",
      },
    });
  }

  const authHeader = req.headers.get("Authorization");
  if (!authHeader) {
    return new Response(JSON.stringify({ error: "Missing Authorization header" }), {
      status: 401,
    });
  }

  // Create a client scoped to the calling user (using their JWT)
  const userClient = createClient(SUPABASE_URL, Deno.env.get("SUPABASE_ANON_KEY")!, {
    global: { headers: { Authorization: authHeader } },
  });

  // Verify the user identity
  const { data: { user }, error: authError } = await userClient.auth.getUser();
  if (authError || !user) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401 });
  }

  // Parse request body
  let ttl = DEFAULT_TTL_MINUTES;
  try {
    const body = await req.json();
    ttl = Math.min(MAX_TTL_MINUTES, Math.max(1, Number(body?.ttl_minutes) || DEFAULT_TTL_MINUTES));
  } catch {
    // Use default TTL
  }

  // Generate pairing code and its hash
  const pairingCode = generatePairingCode();
  const pairingHash = await sha256hex(pairingCode);

  const sessionId = crypto.randomUUID();
  const channelTopic = `browser_session:${sessionId}`;
  const expiresAt = new Date(Date.now() + ttl * 60 * 1000).toISOString();

  // Use service-role key to bypass RLS for the INSERT
  const adminClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  const { error: insertError } = await adminClient.from("browser_input_sessions").insert({
    id: sessionId,
    owner_id: user.id,
    pairing_hash: pairingHash,
    channel_topic: channelTopic,
    expires_at: expiresAt,
    status: "waiting",
  });

  if (insertError) {
    console.error("create-browser-session: insert error", insertError.message);
    return new Response(JSON.stringify({ error: "Session creation failed" }), { status: 500 });
  }

  // Never log the pairing_code
  console.log(`create-browser-session: session=${sessionId} expires=${expiresAt} owner=${user.id}`);

  return new Response(
    JSON.stringify({
      session_id: sessionId,
      pairing_code: pairingCode,
      channel_topic: channelTopic,
      expires_at: expiresAt,
    }),
    {
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    }
  );
});
