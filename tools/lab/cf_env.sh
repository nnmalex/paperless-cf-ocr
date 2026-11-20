# Source me: borrows the `cf` CLI OAuth session (in memory only) for lab runs.
for _i in 1 2 3 4 5; do
  CF_ACCOUNT_ID="$(cf auth whoami 2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["accounts"][0]["id"])
except Exception: pass')"
  [ -n "$CF_ACCOUNT_ID" ] && break; sleep 2
done
export CF_ACCOUNT_ID
export CF_API_TOKEN="$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/Library/Preferences/cloudflare/config/default.json")))["oauth_token"])')"
export CF_AIG_TOKEN="$CF_API_TOKEN" CF_GATEWAY_ID="${CF_GATEWAY_ID:-paperless-cf-ocr-test}"
