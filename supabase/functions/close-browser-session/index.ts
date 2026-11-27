// supabase/functions/close-browser-session/index.ts
//
// Edge Function: close-browser-session
//
// Called by the GhostMic desktop client to terminate an active session.
// Marks the session as 'closed' so no further messages can be sent
// and the sender can no longer use the channel.

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

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

  let sessionId = "";
  try {
    const body = await req.json();
    sessionId = String(body?.session_id || "").trim();
  } catch {
    return jsonResponse({ error: "Invalid request body" }, 400);
  }

  if (!sessionId) {
    return jsonResponse({ error: "session_id is required" }, 400);
  }

  const adminClient = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

  // Only the owner can close a session
  const { error: updateError } = await adminClient
    .from("browser_input_sessions")
    .update({ status: "closed" })
    .eq("id", sessionId)
    .eq("owner_id", user.id)
    .neq("status", "closed");

  if (updateError) {
    console.error("close-browser-session: update error", updateError.message);
    return jsonResponse({ error: "Failed to close session" }, 500);
  }

  console.log(`close-browser-session: session=${sessionId} closed by owner=${user.id}`);

  return jsonResponse({ ok: true });
});
