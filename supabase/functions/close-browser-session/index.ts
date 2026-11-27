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
    return new Response(JSON.stringify({ error: "Missing Authorization header" }), { status: 401 });
  }

  const userClient = createClient(SUPABASE_URL, Deno.env.get("SUPABASE_ANON_KEY")!, {
    global: { headers: { Authorization: authHeader } },
  });

  const { data: { user }, error: authError } = await userClient.auth.getUser();
  if (authError || !user) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401 });
  }

  let sessionId = "";
  try {
    const body = await req.json();
    sessionId = String(body?.session_id || "").trim();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid request body" }), { status: 400 });
  }

  if (!sessionId) {
    return new Response(JSON.stringify({ error: "session_id is required" }), { status: 400 });
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
    return new Response(JSON.stringify({ error: "Failed to close session" }), { status: 500 });
  }

  console.log(`close-browser-session: session=${sessionId} closed by owner=${user.id}`);

  return new Response(JSON.stringify({ ok: true }), {
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
    },
  });
});
