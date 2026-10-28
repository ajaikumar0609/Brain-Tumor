#!/usr/bin/env bash
# Integration smoke test against a RUNNING stack (./app/run_dev.sh). Goes through the Vite proxy on 5173, like the browser does.
set -uo pipefail
B=${1:-http://localhost:5173}; fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
chk "frontend serves the page"            "curl -sf $B/ | grep -q 'NeuroVision AI'"
chk "proxy -> /api/health"                "curl -sf $B/api/health | grep -q '\"ok\"'"
chk "models available"                    "curl -sf $B/api/models/status | python3 -c 'import json,sys;d=json.load(sys.stdin);assert d[\"generator\"][\"available\"] and d[\"classifier\"][\"available\"]'"
chk "metrics: DCGAN FID 154.7"            "curl -sf $B/api/metrics | python3 -c 'import json,sys;d=json.load(sys.stdin);assert round([r for r in d[\"gan\"] if r[\"model\"]==\"DCGAN\"][0][\"fid\"],1)==154.7'"
chk "generate 4 images via proxy"         "curl -sf -X POST $B/api/generate -H 'Content-Type: application/json' -d '{\"class\":\"notumor\",\"count\":4,\"seed\":1}' | python3 -c 'import json,sys;assert len(json.load(sys.stdin)[\"images\"])==4'"
chk "classify rejects a text file"        "curl -s -F file=@/etc/hosts $B/api/classify | grep -q not_an_image"
exit $fail
