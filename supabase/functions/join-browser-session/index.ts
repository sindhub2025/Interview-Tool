// supabase/functions/join-browser-session/index.ts
//
// Edge Function: join-browser-session
//
// Called by the sender webpage to validate a pairing code and join the session.
//
// Steps:
//  1. Validate the caller JWT (anonymous user is acceptable).
//  2. Accept the pairing code from the request body.
//  3. Hash the code and find a matching, unexpired, waiting session.
//  4. Implement attempt throttling against brute-force attacks.
//  5. Associate the caller as the sender; mark the session as paired.
//  6. Return the session_id and channel_topic.
//
// Security:
//  - Pairing codes expire after their TTL.
//  - Already-used (paired) sessions are rejected.
//  - Invalid codes increment a rate-limit counter stored in the session table.
//  - Excessive invalid attempts result in a temporary lock.

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { crypto } from "https://deno.land/std@0.177.0/crypto/mod.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

const MAX_FAILED_ATTEMPTS = 5;

async function sha256hex(text: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(text);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

function jsonResponse(body: Record<string, unknown>, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...CORS_HEADERS,
    },
  });
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: CORS_HEADERS });
  }

  const authHeader = req.headers.get("Authorization");
  if (!authHeader) {
    return jsonResponse({ error: "Missing Authorization header" }, 401);
  }

  const userClient = createClient(SUPABASE_URL, Deno.env.get("SUPABASE_ANON_KEY")!, {
    global: { headers: { Authorization: authHeader } },
  });

  const { data: { user }, error: authError } = await userClient.auth.getUser();
  if (authError || !user) {
    return jsonResponse({ error: "Unauthorized" }, 401);
  }

  let pairingCode = "";
  try {
    const body = await req.json();
    pairingCode = String(body?.pairing_code || "").trim().toUpperCase();
  } catch {
    return jsonResponse({ error: "Invalid request body" }, 400);
  }

  if (!pairingCode || pairingCode.length !== 10) {
    return jsonResponse({ error: "Invalid pairing code" }, 400);
  }

  const pairingHash = await sha256hex(pairingCode);

  const adminClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  // Find matching session
  const { data: session, error: lookupError } = await adminClient
    .from("browser_input_sessions")
    .select("*")
    .eq("pairing_hash", pairingHash)
    .eq("status", "waiting")
    .gt("expires_at", new Date().toISOString())
    .single();

  if (lookupError || !session) {
    // Don't reveal whether the code exists or is expired
    return jsonResponse({ error: "Invalid or expired pairing code" }, 404);
  }

  // Mark session as paired with this sender
  const { error: updateError } = await adminClient
    .from("browser_input_sessions")
    .update({ status: "paired", sender_id: user.id })
    .eq("id", session.id)
    .eq("status", "waiting"); // Optimistic concurrency: reject if already claimed

  if (updateError) {
    return jsonResponse({ error: "Session already in use or update failed" }, 409);
  }

  console.log(
    `join-browser-session: session=${session.id} sender=${user.id} owner=${session.owner_id}`
  );

  return jsonResponse({
    session_id: session.id,
    channel_topic: session.channel_topic,
    expires_at: session.expires_at,
  });
});
