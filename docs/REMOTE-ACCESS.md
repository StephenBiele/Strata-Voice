# Use it from your phone (remote access via Tailscale)

Your Mac does all the work — speech, models, memory. Your phone is just a
microphone and a speaker, connected over [Tailscale](https://tailscale.com)'s
encrypted, private network. Nothing is exposed to the public internet.

## One-time setup

1. Install Tailscale on the Mac and on your phone, signed into the same
   account (that's your "tailnet").
2. On the Mac, with Strata Voice running:

   ```sh
   ./remote.sh on
   ```

   **First time only:** Tailscale asks you to enable Serve for your tailnet —
   the command prints a `login.tailscale.com` link and waits. Open it, approve,
   and the command finishes on its own (it may also enable MagicDNS + HTTPS
   certificates; approve those too).

   It then prints your private URL — something like
   `https://your-mac.tail1234.ts.net`.
3. On the phone (Tailscale connected): open that URL in the browser, then
   **Add to Home Screen**. You get a full-screen app with the Strata Voice
   icon; the mic prompt appears on your first conversation.

`./remote.sh off` stops sharing; `./remote.sh status` shows the current state.

## How it works

- **Tailscale Serve** gives the app a real HTTPS certificate on your tailnet —
  which is what lets the phone's browser use the microphone (browsers refuse
  mic access on plain HTTP).
- The main app is served at the root; the hands-free voice-detection channel
  rides the same origin under `/vadsvc`, so barge-in works remotely too.
- Everything still runs on the Mac: audio is transcribed, answered, and spoken
  there; your memories never leave it. The phone streams audio both ways.

## Honest limitations

- **The Mac must be awake** with the server (or Mac app) running. A
  closed-lid sleeping Mac ends the party — `caffeinate` or Energy Saver
  settings help.
- **Latency** adds roughly 20–80 ms each way on LTE. Conversation works fine;
  it just feels a beat slower than at your desk.
- **No login.** Privacy comes from the tailnet: only devices signed into your
  Tailscale account can reach it. If you share your tailnet with other people,
  they can reach your assistant — and your memories. **Never use `tailscale
  funnel` with this** — that would put your memories on the public internet.
- The screen stays awake during calls (on purpose); a locked phone suspends
  the conversation until you return.
