-- ============================================================
-- Migration: 001_browser_input_sessions.sql
--
-- Creates the browser_input_sessions table for pairing state.
-- Messages are NOT stored permanently; this table only tracks
-- who is authorised to use each Realtime channel.
--
-- Security:
-- - Only the authenticated owner can read their own session.
-- - Only a paired sender can publish to the channel.
-- - RLS is enabled; no policy allows unauthenticated access.
-- - Pairing tokens are stored as SHA-256 hashes only.
-- - Sessions expire after ttl_minutes (default 10).
-- ============================================================

-- Enable pgcrypto for gen_random_uuid() and encode()
create extension if not exists pgcrypto;

-- ── Sessions table ──────────────────────────────────────────
create table if not exists public.browser_input_sessions (
    id              uuid        primary key default gen_random_uuid(),
    -- The authenticated Supabase user who created (owns) this session.
    owner_id        uuid        not null references auth.users(id) on delete cascade,
    -- SHA-256 hex hash of the plaintext pairing code.  Never stored plain.
    pairing_hash    text        not null,
    -- The Realtime channel topic (e.g. "browser_session:<id>").
    channel_topic   text        not null unique,
    -- When this session expires.
    expires_at      timestamptz not null,
    -- The Supabase user who successfully paired as a sender (null until paired).
    sender_id       uuid        references auth.users(id) on delete set null,
    -- Lifecycle state: waiting | paired | closed
    status          text        not null default 'waiting'
                    check (status in ('waiting','paired','closed')),
    created_at      timestamptz not null default now()
);

-- Index to accelerate expiry cleanup and status lookups
create index if not exists browser_input_sessions_owner_idx
    on public.browser_input_sessions (owner_id);

create index if not exists browser_input_sessions_expires_idx
    on public.browser_input_sessions (expires_at);

-- ── Row-Level Security ──────────────────────────────────────
alter table public.browser_input_sessions enable row level security;

-- Owner can view their own sessions
create policy "owner_select"
    on public.browser_input_sessions
    for select
    using (owner_id = auth.uid());

-- Owner can insert their own sessions (Edge Function runs as owner via JWT)
create policy "owner_insert"
    on public.browser_input_sessions
    for insert
    with check (owner_id = auth.uid());

-- Owner can update status of their own sessions (e.g. to close)
create policy "owner_update"
    on public.browser_input_sessions
    for update
    using (owner_id = auth.uid());

-- Sender can read the session once they know the topic (validated in Edge Function)
create policy "sender_select_joined"
    on public.browser_input_sessions
    for select
    using (sender_id = auth.uid());

-- ── Realtime channel authorisation helpers ──────────────────
-- These SECURITY DEFINER functions are called by Realtime RLS policies.
-- They use a fixed search_path to prevent search-path injection.

create or replace function public.is_session_owner(p_topic text)
returns boolean
language sql
security definer
set search_path = public
stable
as $$
    select exists (
        select 1
        from public.browser_input_sessions s
        where s.channel_topic = p_topic
          and s.owner_id = auth.uid()
          and s.status != 'closed'
          and s.expires_at > now()
    );
$$;

create or replace function public.is_session_sender(p_topic text)
returns boolean
language sql
security definer
set search_path = public
stable
as $$
    select exists (
        select 1
        from public.browser_input_sessions s
        where s.channel_topic = p_topic
          and s.sender_id = auth.uid()
          and s.status = 'paired'
          and s.expires_at > now()
    );
$$;

-- ── Realtime channel policies ───────────────────────────────
-- Note: Supabase manages realtime.messages internally.
-- We control access through the channel-level private config
-- and the Edge Function that sets the JWT claims / channel grants.
-- The functions above are referenced by the create-browser-session
-- and join-browser-session Edge Functions when issuing channel tokens.

-- ── Scheduled cleanup ──────────────────────────────────────
-- (Optional) pg_cron job to close expired sessions every minute.
-- Uncomment if pg_cron is enabled in your Supabase project.
--
-- select cron.schedule(
--     'browser-session-cleanup',
--     '* * * * *',
--     $$
--         update public.browser_input_sessions
--         set status = 'closed'
--         where status != 'closed'
--           and expires_at <= now();
--     $$
-- );
