# Supabase Setup Guide for GhostMic Browser Input

This guide walks through setting up the Supabase backend required for GhostMic's **Browser Input** feature.

## Architecture Notes

- **No message persistence**: Messages are transmitted exclusively over private authenticated **Supabase Realtime Broadcast** channels and are never written to any database table.
- **Session metadata only**: Only temporary pairing session metadata (`id`, `pairing_code_hash`, `owner_user_id`, `sender_user_id`, `channel_name`, `status`, `expires_at`) is stored in `public.browser_input_sessions`.
- **Pairing code hashing**: Plaintext 10-character Crockford Base32 pairing codes are never stored in the database; only their SHA-256 hash is persisted.

---

## 1. Create a Supabase Project

1. Go to [https://supabase.com](https://supabase.com) and create a new project.
2. Note your **Project Reference ID** and **Project URL** (`https://<project-ref>.supabase.co`).
3. Go to **Project Settings -> API** and copy the **anon / publishable** key.
   - **Do NOT** put the `service_role` key into GhostMic or `browser_sender`.

## 2. Enable Anonymous Sign-Ins

Both GhostMic (Laptop 1) and the React sender webpage (Laptop 2) use Supabase Anonymous Authentication so that Realtime channels can be private and authenticated without requiring user accounts.

1. In the Supabase Dashboard, go to **Authentication -> Providers**.
2. Enable **Anonymous Sign-Ins** (under User Signups / Providers -> Anonymous).
3. Save changes.

## 3. Apply the Database Migration

Apply `supabase/migrations/001_browser_input_sessions.sql` to your Supabase project:

**Option A: Supabase SQL Editor**
1. Open **SQL Editor** in the Supabase Dashboard.
2. Paste the contents of `supabase/migrations/001_browser_input_sessions.sql`.
3. Click **Run**.

**Option B: Supabase CLI**
```bash
supabase link --project-ref <your-project-ref>
supabase db push
```

This creates:
- `public.browser_input_sessions` table with Row Level Security (RLS) enabled.
- `public.is_session_owner(text)` and `public.is_session_sender(text)` helper functions.
- Realtime `realtime.messages` RLS policies restricting private broadcast channel read/write access to the paired owner and sender of an active session.

## 4. Deploy the Edge Functions

Install the Supabase CLI if needed, then deploy the three Edge Functions from the repository root:

```bash
supabase login
supabase link --project-ref <your-project-ref>

supabase functions deploy create-browser-session
supabase functions deploy join-browser-session
supabase functions deploy close-browser-session
```

> **Note**: Edge Functions automatically receive `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `SUPABASE_SERVICE_ROLE_KEY` in the Supabase hosted environment. If you need to verify or override secrets:
> ```bash
> supabase secrets set SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
> ```

## 5. Configure GhostMic (Laptop 1)

1. Launch GhostMic.
2. Open **Settings -> Browser Input**.
3. Fill in:
   - **Enable remote Browser Input panel**: Checked
   - **Supabase URL**: `https://<your-project-ref>.supabase.co`
   - **Client API Key**: Your Supabase `anon` / publishable key
   - **Sender Webpage URL**: The URL where `browser_sender` is hosted (e.g. `https://your-sender.netlify.app`)
4. Click **OK** and restart GhostMic (or start a session from the **Remote** panel).

## 6. Deploy & Configure the Sender Webpage (Laptop 2)

See [browser_sender/README.md](browser_sender/README.md) for full build and deployment instructions.
