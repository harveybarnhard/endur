"""Strava sign-in for running the Manhattan pipeline on your own machine.

    python endur/manhattan/strava_auth.py login   # once: sign in through the browser

Keeps the app's Client ID and Secret plus the athlete's refresh token in
~/.config/endur-manhattan/strava.json (mode 0600, outside every repo). Runs then
trade the refresh token for a short-lived access token by themselves.
"""
import getpass
import http.server
import json
import os
import sys
import time
import urllib.parse
import webbrowser

import requests

CREDS = os.path.expanduser("~/.config/endur-manhattan/strava.json")
OAUTH = "https://www.strava.com/oauth"
PORT = 8723
REDIRECT = f"http://localhost:{PORT}/exchange_token"
SCOPE = "read,activity:read_all"


def save(path, creds):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(creds, f, indent=1)
    os.replace(tmp, path)


def load(path):
    with open(path) as f:
        return json.load(f)


def _token_request(creds, **data):
    r = requests.post(f"{OAUTH}/token", timeout=60, data={
        "client_id": creds["client_id"], "client_secret": creds["client_secret"], **data})
    if r.status_code >= 400:
        refresh = data.get("grant_type") == "refresh_token"
        hint = " Sign in again: endur/manhattan/local.sh login" if refresh else ""
        sys.exit(f"Strava refused the request ({r.status_code}): {r.text[:200]}.{hint}")
    tok = r.json()
    creds.update(access_token=tok["access_token"], refresh_token=tok["refresh_token"],
                 expires_at=tok["expires_at"])
    return tok


def access_token(path=CREDS, force=False):
    """A valid access token, refreshed (and saved) when it expires within 5 minutes, or when
    `force`d after Strava rejected it. Refreshes only when needed: every refresh is a chance
    for Strava to replace the refresh token, which the GitHub Action also holds."""
    try:
        creds = load(path)
    except FileNotFoundError:
        sys.exit(f"No Strava sign-in at {path}. Run: endur/manhattan/local.sh login")
    if force or creds.get("expires_at", 0) < time.time() + 300:
        old = creds["refresh_token"]
        _token_request(creds, grant_type="refresh_token", refresh_token=old)
        save(path, creds)
        if creds["refresh_token"] != old:
            print("NOTE: Strava issued a new refresh token (saved locally). The GitHub Action holds the\n"
                  "old one in data/strava_tokens.json.gpg, which may now stop working. If its next run\n"
                  "fails, move the Action to this sign-in (Client ID/Secret + refresh-token secrets).",
                  file=sys.stderr, flush=True)
    return creds["access_token"]


def _catch_code():
    """Serve one request on localhost and return the query parameters Strava redirects with."""
    got = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            got.update({k: v[0] for k, v in q.items()})
            ok = "code" in got
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            msg = "Signed in. You can close this tab." if ok else "Sign-in was cancelled."
            self.wfile.write(f"<p style='font:18px sans-serif;margin:3em'>{msg}</p>".encode())

        def log_message(self, *a):
            pass

    class Server(http.server.HTTPServer):
        timed_out = False

        def handle_timeout(self):
            self.timed_out = True

    with Server(("127.0.0.1", PORT), Handler) as srv:
        srv.timeout = 300  # give up after 5 idle minutes
        while "code" not in got and "error" not in got and not srv.timed_out:
            srv.handle_request()
    return got


def login(path=CREDS):
    try:
        creds = load(path)
        print(f"Using the saved Client ID {creds['client_id']} (delete {path} to enter new ones).")
    except FileNotFoundError:
        print("From https://www.strava.com/settings/api (\"My API Application\"):")
        cid = input("  Client ID: ").strip()
        secret = getpass.getpass("  Client Secret (hidden as you type): ").strip()
        if not cid.isdigit() or not secret:
            sys.exit("That doesn't look right: the Client ID is a number and the secret a long hex string.")
        creds = {"client_id": cid, "client_secret": secret}
    url = f"{OAUTH}/authorize?" + urllib.parse.urlencode({
        "client_id": creds["client_id"], "response_type": "code", "redirect_uri": REDIRECT,
        "approval_prompt": "auto", "scope": SCOPE})
    print("\nOpening Strava in your browser; click Authorize. If nothing opens, visit:\n  " + url + "\n")
    webbrowser.open(url)
    try:
        got = _catch_code()
    except OSError:  # port busy: fall back to pasting the redirected address
        got = {}
    if "code" not in got and "error" not in got:
        pasted = input("Paste the address your browser ended up on (starts with http://localhost): ").strip()
        got = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(pasted).query).items()}
    if "code" not in got:
        sys.exit(f"Sign-in didn't complete ({got.get('error', 'no code returned')}).")
    if "activity:read_all" not in got.get("scope", ""):
        sys.exit("Strava didn't grant access to activities. Run login again and leave "
                 "\"View data about your private activities\" ticked.")
    tok = _token_request(creds, grant_type="authorization_code", code=got["code"])
    save(path, creds)
    who = tok.get("athlete") or {}
    name = " ".join(filter(None, [who.get("firstname"), who.get("lastname")])) or "you"
    print(f"Signed in as {name}. Saved to {path.replace(os.path.expanduser('~'), '~')}")


if __name__ == "__main__":
    if sys.argv[1:] != ["login"]:
        sys.exit(__doc__)
    login()
