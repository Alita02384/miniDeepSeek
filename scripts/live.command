#!/usr/bin/env bash
# Double-click this file (macOS) to open the live training dashboard.
#
# A browser cannot open an SSH connection, so *something* has to run on this
# machine and hold one.  That is what scripts/live_dashboard.py is: it follows
# the remote metrics.jsonl and serves the page on 127.0.0.1.  This launcher
# removes everything else you would otherwise have to type -- host, port,
# password, the browser tab.
#
# First run: asks for the SSH password once and caches it in
# ~/.config/minideepseek/ssh_pass (mode 600, outside the repo, never committed).
# Every later run: no prompts.  If you set up a key instead
# (`ssh-copy-id -p 30136 root@<host>`), the password file is not used at all.
#
# Override anything with the environment:
#   MINIDEEPSEEK_SSH_HOST  MINIDEEPSEEK_SSH_PORT  MINIDEEPSEEK_REMOTE_ROOT
#   MINIDEEPSEEK_STAGE     MINIDEEPSEEK_LOCAL_PORT  MINIDEEPSEEK_NO_BROWSER=1
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${MINIDEEPSEEK_SSH_HOST:-hb01-ssh.gpuhome.cc}"
PORT="${MINIDEEPSEEK_SSH_PORT:-30136}"
PASSFILE="${HOME}/.config/minideepseek/ssh_pass"

if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
        -o ConnectTimeout=8 -p "$PORT" "root@${HOST}" true 2>/dev/null; then
  if [ -z "${MINIDEEPSEEK_SSH_PASSWORD:-}" ]; then
    if [ -s "$PASSFILE" ]; then
      MINIDEEPSEEK_SSH_PASSWORD="$(cat "$PASSFILE")"
    else
      mkdir -p "$(dirname "$PASSFILE")"
      chmod 700 "$(dirname "$PASSFILE")"
      printf 'SSH password for root@%s: ' "$HOST"
      read -r -s password
      printf '\n'
      printf '%s' "$password" > "$PASSFILE"
      chmod 600 "$PASSFILE"
      echo "[live] cached in ${PASSFILE} (chmod 600) — delete it to be prompted again"
      MINIDEEPSEEK_SSH_PASSWORD="$password"
    fi
    export MINIDEEPSEEK_SSH_PASSWORD
  fi
fi

if [ ! -x .venv/bin/python ]; then
  echo "no .venv here; run scripts/setup_env.sh first" >&2
  exit 1
fi

exec .venv/bin/python scripts/live_dashboard.py \
  --host "$HOST" --port "$PORT" \
  --remote-root "${MINIDEEPSEEK_REMOTE_ROOT:-/root/rivermind-data/miniDeepSeek}" \
  --stage "${MINIDEEPSEEK_STAGE:-checkpoints/curriculum_145m_3b}"
