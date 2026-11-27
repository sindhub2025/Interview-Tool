-- ============================================================
-- Migration: 002_realtime_messages_policies.sql
--
-- Adds Row Level Security (RLS) policies on realtime.messages
-- for private authenticated Browser Input broadcast channels.
-- ============================================================

create policy "browser_session_read_broadcast"
    on "realtime"."messages"
    for select
    to authenticated
    using (
        realtime.messages.extension = 'broadcast'
        and (
            public.is_session_owner((select realtime.topic()))
            or public.is_session_sender((select realtime.topic()))
        )
    );

create policy "browser_session_write_broadcast"
    on "realtime"."messages"
    for insert
    to authenticated
    with check (
        realtime.messages.extension = 'broadcast'
        and (
            public.is_session_owner((select realtime.topic()))
            or public.is_session_sender((select realtime.topic()))
        )
    );
