#!/bin/sh
# pre-commit's pre-push stage passes the refs in PRE_COMMIT_* variables, not on stdin: rebuild git's stdin line
# ("<local ref> <local sha> <remote ref> <remote sha>") and hand it to the guard.
printf '%s %s %s %s\n' "${PRE_COMMIT_LOCAL_BRANCH:-HEAD}" "${PRE_COMMIT_TO_REF:?}" \
  "${PRE_COMMIT_REMOTE_BRANCH:-}" "${PRE_COMMIT_FROM_REF:-0000000000000000000000000000000000000000}" |
  exec python tools/hooks/privacy_guard.py push "${PRE_COMMIT_REMOTE_NAME:-}" "${PRE_COMMIT_REMOTE_URL:-}"
