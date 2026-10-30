#!/usr/bin/env bash
# ==============================================================================
# trigger_github_build.sh
# 
# Automatically trigger a fresh Windows & Linux standalone executable build
# on GitHub Actions, monitors its progress live, and downloads the artifacts.
# ==============================================================================

set -euo pipefail

REPO="eiktdmo/agymolden"
WORKFLOW_ID="402209647"
BRANCH="${1:-main}"

# Default token or read from environment variable GITHUB_TOKEN
GITHUB_TOKEN="${GITHUB_TOKEN:-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}"

echo "========================================================"
echo "  AgyMolden - Trigger Remote GitHub Standalone Build"
echo "  Repository: ${REPO}"
echo "  Target Branch: ${BRANCH}"
echo "========================================================"

# 1. Trigger the workflow
echo "[1/4] Triggering GitHub Actions workflow..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  -H "Content-Type: application/json" \
  "https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW_ID}/dispatches" \
  -d "{\"ref\":\"${BRANCH}\"}")

if [ "$HTTP_STATUS" -ne 204 ]; then
  echo "[-] Failed to trigger workflow (HTTP Status: $HTTP_STATUS)."
  exit 1
fi
echo "[+] Workflow successfully dispatched!"

# 2. Wait for GitHub to register the new run
echo "[2/4] Locating new workflow run..."
sleep 4
RUN_INFO=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
  "https://api.github.com/repos/${REPO}/actions/runs?per_page=1")

RUN_ID=$(echo "$RUN_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin)['workflow_runs'][0]['id'])")
RUN_URL=$(echo "$RUN_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin)['workflow_runs'][0]['html_url'])")

echo "[+] Active Run ID: ${RUN_ID}"
echo "[+] Live Web Dashboard: ${RUN_URL}"
echo ""

# 3. Poll build status until complete
echo "[3/4] Building standalone binaries on Windows and Linux runners..."
while true; do
  STATUS_INFO=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
    "https://api.github.com/repos/${REPO}/actions/runs/${RUN_ID}")
  
  STATUS=$(echo "$STATUS_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status'))")
  CONCLUSION=$(echo "$STATUS_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin).get('conclusion'))")

  if [ "$STATUS" = "completed" ]; then
    echo ""
    echo "[+] Build finished with conclusion: ${CONCLUSION}"
    if [ "$CONCLUSION" != "success" ]; then
      echo "[-] The build did not succeed. Check the logs at: ${RUN_URL}"
      exit 1
    fi
    break
  fi

  echo -n "."
  sleep 15
done

# 4. Fetch and download generated artifacts
echo ""
echo "[4/4] Fetching download links for executable bundles..."
mkdir -p dist/github_builds

ARTIFACTS_JSON=$(curl -s -H "Authorization: token ${GITHUB_TOKEN}" \
  "https://api.github.com/repos/${REPO}/actions/runs/${RUN_ID}/artifacts")

python3 - <<EOF
import json, urllib.request, os

token = "${GITHUB_TOKEN}"
data = json.loads('''${ARTIFACTS_JSON}''')
artifacts = data.get('artifacts', [])

if not artifacts:
    print("[-] No artifacts found.")
else:
    for art in artifacts:
        name = art['name']
        download_url = art['archive_download_url']
        out_file = f"dist/github_builds/{name}.zip"
        print(f"[+] Downloading {name} -> {out_file}...")
        
        req = urllib.request.Request(download_url, headers={
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github+json'
        })
        with urllib.request.urlopen(req) as resp, open(out_file, 'wb') as f:
            f.write(resp.read())
        print(f"    Saved: {out_file}")

print("\n========================================================")
print("  Build complete! Files are saved in: dist/github_builds/")
print("========================================================")
EOF
