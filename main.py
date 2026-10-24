-- =====================================================================
-- MythosAI v2.0  -  Supabase schema
-- Run the whole file once in: Supabase Dashboard -> SQL Editor.
-- Safe to re-run (everything is idempotent).
--
-- DAILY REFILL: the source of truth is a lazy refill inside consume_token()
-- (the first request after midnight resets the balance), so it works even if
-- the cron job below is never enabled. The pg_cron job just keeps the stored
-- balances tidy for dashboards / support.
-- =====================================================================

-- 1. PROFILES ----------------------------------------------------------
create table if not exists public.profiles (
  id               uuid primary key references auth.users (id) on delete cascade,
  token_balance    integer     not null default 20 check (token_balance >= 0),
  last_refill_date date        not null default (now() at time zone 'utc')::date,
  created_at       timestamptz not null default now()
);

-- If you already had a profiles table, make sure it has the two columns we need:
alter table public.profiles add column if not exists token_balance    integer not null default 20;
alter table public.profiles add column if not exists last_refill_date date    not null default (now() at time zone 'utc')::date;

-- Players may READ their own profile. Nobody but the server (service_role) can write it,
-- so a player can never top up their own tokens from the browser.
alter table public.profiles enable row level security;
drop policy if exists "profiles: read own" on public.profiles;
create policy "profiles: read own" on public.profiles
  for select to authenticated using (auth.uid() = id);

-- Every new sign-up gets a profile automatically ...
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id) values (new.id) on conflict (id) do nothing;
  return new;
end $$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ... and users who signed up before this trigger existed are back-filled now.
insert into public.profiles (id) select id from auth.users on conflict (id) do nothing;


-- 2. ATOMIC TOKEN FUNCTIONS (called by the server through supabase.rpc) --
-- consume_token: ONE statement = lazy midnight refill + decrement + guard.
-- Row-level locking makes concurrent requests safe: two calls racing for the
-- last token cannot both win. Returns the new balance, or NULL if the player
-- is out of tokens (or has no profile).
create or replace function public.consume_token(
  p_user_id     uuid,
  p_daily_limit integer default 20,
  p_tz          text    default 'UTC'
) returns integer
language plpgsql security definer set search_path = public as $$
declare
  v_today   date := (now() at time zone p_tz)::date;
  v_balance integer;
begin
  update public.profiles
     set token_balance    = case when last_refill_date < v_today
                                 then p_daily_limit - 1          -- new day: refill, then spend one
                                 else token_balance - 1 end,
         last_refill_date = v_today
   where id = p_user_id
     and (last_refill_date < v_today or token_balance > 0)
  returning token_balance into v_balance;

  return v_balance;   -- NULL => no row matched => out of tokens
end $$;

-- refund_token: gives one token back when a turn fails AFTER it was charged
-- (LLM down, DB error ...). Never exceeds the daily limit.
create or replace function public.refund_token(
  p_user_id     uuid,
  p_daily_limit integer default 20
) returns integer
language plpgsql security definer set search_path = public as $$
declare v_balance integer;
begin
  update public.profiles
     set token_balance = least(token_balance + 1, p_daily_limit)
   where id = p_user_id
  returning token_balance into v_balance;
  return v_balance;
end $$;

-- refill_all_tokens: used by the midnight cron job. Only touches profiles that
-- have not been refilled today, so it can never double-refill a player.
create or replace function public.refill_all_tokens(
  p_daily_limit integer default 20,
  p_tz          text    default 'UTC'
) returns integer
language plpgsql security definer set search_path = public as $$
declare
  v_today date := (now() at time zone p_tz)::date;
  v_rows  integer;
begin
  update public.profiles
     set token_balance = p_daily_limit, last_refill_date = v_today
   where last_refill_date < v_today;
  get diagnostics v_rows = row_count;
  return v_rows;
end $$;

-- SECURITY: these must NOT be callable from the browser (anon key), otherwise
-- anyone could refill or drain tokens. Only the server's service_role may call them.
revoke all on function public.consume_token(uuid, integer, text)  from public, anon, authenticated;
revoke all on function public.refund_token(uuid, integer)         from public, anon, authenticated;
revoke all on function public.refill_all_tokens(integer, text)    from public, anon, authenticated;
grant execute on function public.consume_token(uuid, integer, text)  to service_role;
grant execute on function public.refund_token(uuid, integer)         to service_role;
grant execute on function public.refill_all_tokens(integer, text)    to service_role;


-- 3. GAME SESSIONS (server-side memory; serverless functions keep nothing in RAM)
create table if not exists public.game_sessions (
  id                 uuid primary key default gen_random_uuid(),
  user_id            uuid        not null references auth.users (id) on delete cascade,
  scene_id           text,                                -- scene of the previous turn (NULL = brand-new campaign)
  hot_buffer         jsonb       not null default '[]',   -- last N turns of the CURRENT scene, verbatim (flushed on scene change)
  turn_log           jsonb       not null default '[]',   -- turns still waiting for a warm summary (survives a killed background task)
  warm_summary       text        not null default '',     -- rolling "Scene History Summary", one line per summarised chunk
  summarized_through integer     not null default 0,      -- highest turn seq already folded into warm_summary (written ONLY by the summariser)
  turn_count         integer     not null default 0,
  version            integer     not null default 0,      -- optimistic lock: stops double-tap turns from clobbering each other
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now()
);
create index if not exists game_sessions_user_recent on public.game_sessions (user_id, updated_at desc);

-- Server-only table: RLS on with NO policies = browsers can't touch it; service_role bypasses RLS.
alter table public.game_sessions enable row level security;


-- 4. MIDNIGHT REFILL JOB (optional but recommended) --------------------
-- Enable the extension first: Dashboard -> Database -> Extensions -> pg_cron.
-- pg_cron runs on the database clock, which is UTC on Supabase.
-- If REFILL_TIMEZONE in your .env is not UTC, shift the hour so the job fires at
-- that zone's midnight, and pass the same zone + limit as arguments below.
create extension if not exists pg_cron;
select cron.schedule(
  'mythosai-midnight-refill',
  '0 0 * * *',
  $$ select public.refill_all_tokens(20, 'UTC') $$
);
-- To remove it later:  select cron.unschedule('mythosai-midnight-refill');


-- 5. COLD MEMORY (OPTIONAL - only when you switch COLD_MEMORY_BACKEND=pgvector)
-- Uncomment, pick the vector size that matches your embedding model, and wire an
-- embedder into cold_memory.get_cold_memory(). Until then the mock backend is used.
--
-- create extension if not exists vector;
--
-- create table if not exists public.campaign_milestones (
--   id         bigint generated always as identity primary key,
--   session_id uuid not null references public.game_sessions (id) on delete cascade,
--   content    text not null,
--   tags       text[] not null default '{}',
--   embedding  vector(384) not null,          -- 384 = e.g. gte-small / all-MiniLM-L6-v2
--   created_at timestamptz not null default now()
-- );
-- alter table public.campaign_milestones enable row level security;
-- create index if not exists campaign_milestones_embedding
--   on public.campaign_milestones using hnsw (embedding vector_cosine_ops);
--
-- create or replace function public.match_milestones(
--   p_session_id uuid, query_embedding vector(384), match_count int default 3, min_similarity float default 0.35
-- ) returns table (content text, tags text[], similarity float)
-- language sql stable as $$
--   select m.content, m.tags, 1 - (m.embedding <=> query_embedding) as similarity
--     from public.campaign_milestones m
--    where m.session_id = p_session_id
--      and 1 - (m.embedding <=> query_embedding) >= min_similarity
--    order by m.embedding <=> query_embedding
--    limit match_count
-- $$;
-- revoke all on function public.match_milestones(uuid, vector, int, float) from public, anon, authenticated;
-- grant execute on function public.match_milestones(uuid, vector, int, float) to service_role;


-- Make PostgREST (the Supabase API) see the new functions/tables immediately.
notify pgrst, 'reload schema';
