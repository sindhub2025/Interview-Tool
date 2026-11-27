# GhostMic Browser Sender

A standalone React + Vite + TypeScript web application for **Laptop 2 (Sender)** to manually send questions, code snippets, and answers to **GhostMic (Laptop 1)** over a private Supabase Realtime Broadcast channel.

## Features

- Zero-install: runs in any modern web browser.
- Pairing-code authentication using temporary 10-character Crockford Base32 codes.
- Supports `question`, `code` (with language selector), and `answer` message types.
- Enforces the 100 KB (`102400` bytes) payload limit before sending.
- Stores no messages in browser localStorage, databases, or Supabase tables.

## Environment Variables

Copy `.env.example` to `.env.local` for local development or configure these variables in your hosting provider:

```env
VITE_SUPABASE_URL=https://<your-project-ref>.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=<your-supabase-anon-or-publishable-key>
```

> **Security**: Never use a `service_role` or secret key in `VITE_*` variables. Only the public client/anon key is used.

## Local Development

```bash
cd browser_sender
npm install
npm run dev
```

## Production Build

```bash
cd browser_sender
npm install
npm run build
```

The production static bundle is output to `browser_sender/dist/`.

## Deployment

### Option 1: Netlify

**Using Netlify CLI:**
```bash
cd browser_sender
npm install
npm run build
npx netlify-cli deploy --prod --dir=dist
```

**Using Git Continuous Deployment:**
1. Connect your repository in Netlify.
2. Set **Base directory** to `browser_sender`.
3. Set **Build command** to `npm run build`.
4. Set **Publish directory** to `browser_sender/dist`.
5. Add environment variables `VITE_SUPABASE_URL` and `VITE_SUPABASE_PUBLISHABLE_KEY` in Site Settings -> Environment Variables.

### Option 2: GitHub Pages

1. Update `base` in `browser_sender/vite.config.ts` if deploying to a repository subpath (e.g. `base: "/Interview-Tool/"`).
2. Build with environment variables set:
   ```bash
   cd browser_sender
   npm install
   VITE_SUPABASE_URL=https://<project-ref>.supabase.co \
   VITE_SUPABASE_PUBLISHABLE_KEY=<anon-key> \
   npm run build
   ```
3. Publish the `browser_sender/dist/` directory to your `gh-pages` branch or GitHub Pages workflow.

### Post-Deployment

Copy the deployed sender URL (e.g. `https://your-site.netlify.app`) into GhostMic **Settings -> Browser Input -> Sender Webpage URL** (or `browser_bridge.sender_url` in `ghostmic/config.json`).
