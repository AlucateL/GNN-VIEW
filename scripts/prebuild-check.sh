#!/usr/bin/env bash
#
# prebuild-check.sh — reproduce the GitLab CI image build locally, before pushing.
#
# docker-compose.yml builds `target: base` (dev server, source mounted as a volume).
# .gitlab-ci.yml builds `--target production` (source COPYd in, next build, no reload).
# They are different images, which is why some failures only ever appear in CI.
#
# This script builds the production targets exactly as CI does, boots them with the
# environment Helm injects in the cluster, and checks that /health answers inside the
# probe budget declared in helm/values.yaml.
#
# Usage:
#   scripts/prebuild-check.sh                  # both layers, CI architecture (amd64)
#   scripts/prebuild-check.sh --layer backend  # one layer only
#   scripts/prebuild-check.sh --native         # host arch — much faster, less faithful
#   scripts/prebuild-check.sh --build-only     # skip the boot/health phase
#   scripts/prebuild-check.sh --keep           # leave containers up for manual poking
#   scripts/prebuild-check.sh --fresh          # --no-cache: actually re-run lint/build
#
# Note: Docker reuses cached layers, so a repeat run can report success without
# re-running `yarn lint && yarn build` or reinstalling Python deps. Use --fresh
# when you need the build genuinely re-verified.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Configuration ────────────────────────────────────────────────────────────
# Mirrors .gitlab-ci.yml (build target, Dockerfile paths) and helm/values.yaml
# (ports, health paths, probe budget, Postgres identity).

PLATFORM="linux/amd64"          # CI runners are x86_64; override with --native
LAYERS=("backend" "frontend")
BUILD_ONLY=0
KEEP=0
FRESH=0

BACKEND_PORT=8081               # 8081/3001 so a running docker-compose stack
FRONTEND_PORT=3001              # (8080/3000) does not collide with this check

# Local boot check only — k8s uses startupProbe (10 min) then liveness 30s×5.
LIVENESS_BUDGET=60
READINESS_BUDGET=40

NET="gnnview-prebuild-net"
PG_NAME="gnnview-prebuild-pg"
BE_NAME="gnnview-prebuild-backend"
FE_NAME="gnnview-prebuild-frontend"

# Cluster Postgres identity (.gitlab-ci.yml POSTGRES_DB / POSTGRES_USER), not the
# graph_db/postgres/postgres that docker-compose.yml uses.
PG_DB="app"
PG_USER="app"
PG_PASS="app"

# Public hostnames from .gitlab-ci.yml — CORS is the classic prod-only failure.
URL_FRONTEND="gnn-view.mooo.com"
URL_BACKEND="gnn-view-api.mooo.com"

# ── Output helpers ───────────────────────────────────────────────────────────
if [ -t 1 ]; then
  BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
  BLUE=$'\033[34m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
  BOLD=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; DIM=""; RESET=""
fi

step() { printf "\n%s==>%s %s%s%s\n" "$BLUE" "$RESET" "$BOLD" "$1" "$RESET"; }
ok()   { printf "  %s✓%s %s\n" "$GREEN" "$RESET" "$1"; }
warn() { printf "  %s!%s %s\n" "$YELLOW" "$RESET" "$1"; WARNINGS=$((WARNINGS + 1)); }
fail() { printf "  %s✗%s %s\n" "$RED" "$RESET" "$1"; FAILURES=$((FAILURES + 1)); }
info() { printf "  %s%s%s\n" "$DIM" "$1" "$RESET"; }

WARNINGS=0
FAILURES=0

# ── Argument parsing ─────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
  case "$1" in
    --native)     PLATFORM="" ;;
    --build-only) BUILD_ONLY=1 ;;
    --keep)       KEEP=1 ;;
    --fresh)      FRESH=1 ;;
    --layer)
      shift
      case "${1:-}" in
        backend|frontend) LAYERS=("$1") ;;
        both)             LAYERS=("backend" "frontend") ;;
        *) echo "--layer expects: backend | frontend | both" >&2; exit 2 ;;
      esac
      ;;
    -h|--help)
      sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# macOS ships bash 3.2, where "${arr[@]}" on an empty array trips `set -u`.
# Expand through the ${arr[@]+...} guard everywhere this array is used.
PLATFORM_ARGS=()
[ -n "$PLATFORM" ] && PLATFORM_ARGS=(--platform "$PLATFORM")

CACHE_ARGS=()
[ "$FRESH" -eq 1 ] && CACHE_ARGS=(--no-cache)

# ── Cleanup ──────────────────────────────────────────────────────────────────
cleanup() {
  if [ "$KEEP" -eq 1 ]; then
    printf "\n%s--keep set: containers left running.%s\n" "$DIM" "$RESET"
    printf "  backend  http://localhost:%s/health\n" "$BACKEND_PORT"
    printf "  frontend http://localhost:%s/\n" "$FRONTEND_PORT"
    printf "  tear down: docker rm -f %s %s %s && docker network rm %s\n" \
      "$BE_NAME" "$FE_NAME" "$PG_NAME" "$NET"
    return
  fi
  docker rm -f "$BE_NAME" "$FE_NAME" "$PG_NAME" >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ── Preflight ────────────────────────────────────────────────────────────────
step "Preflight"

if ! docker info >/dev/null 2>&1; then
  fail "Docker daemon is not running — start Docker Desktop and retry."
  exit 1
fi
ok "Docker daemon reachable"

if [ -n "$PLATFORM" ]; then
  info "Building for $PLATFORM (CI architecture). Host is $(uname -m)."
  [ "$(uname -m)" = "arm64" ] && info "Emulated via QEMU — slow, and startup timings are not representative."
else
  info "Building for host architecture ($(uname -m)) — fast, but does not catch amd64-only wheel issues."
fi

if [ "$FRESH" -eq 1 ]; then
  info "--fresh: cache disabled, lint/build/pip install will genuinely re-run."
else
  info "Docker layer cache in use — a CACHED lint/build step was not re-verified. Use --fresh to force it."
fi

# CI builds from a clean git clone; this script builds from the working tree. Files
# that exist here but are not committed will be missing in CI (and vice versa).
UNTRACKED=$(git ls-files --others --exclude-standard \
  backend/src backend/data frontend/nextjs 2>/dev/null | grep -v '__pycache__' | head -5 || true)
if [ -n "$UNTRACKED" ]; then
  warn "Untracked files inside build context — present locally, MISSING in CI:"
  printf '      %s\n' $UNTRACKED
else
  ok "No untracked files in the build context"
fi

# CI checks out the committed submodule pointer, not your local checkout.
if [ -n "$(git submodule status --recursive | grep -E '^\+' || true)" ]; then
  warn "Submodule pointer differs from the committed one — CI will build different neural-lam source."
else
  ok "Submodule pointer matches the commit"
fi

# ── Build ────────────────────────────────────────────────────────────────────
for layer in "${LAYERS[@]}"; do
  step "Build: $layer (--target production)"
  build_start=$SECONDS

  # Same target, Dockerfile and context as build:buildah in .gitlab-ci.yml.
  # macOS ships bash 3.2, where "${arr[@]}" on an empty array trips `set -u`,
  # so PLATFORM_ARGS is expanded through the ${arr[@]+...} guard here and below.
  if docker build ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} \
      ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} \
      --target production \
      --file "./${layer}/Dockerfile" \
      --tag "gnn-view-${layer}:prebuild" \
      "./${layer}"; then
    ok "$layer image built in $((SECONDS - build_start))s"
  else
    fail "$layer production build FAILED — this is what CI would hit."
    exit 1
  fi
done

if [ "$BUILD_ONLY" -eq 1 ]; then
  step "Summary"
  ok "Build-only mode: images built, boot phase skipped."
  exit 0
fi

# ── Boot ─────────────────────────────────────────────────────────────────────
step "Boot: cluster-like environment"

docker rm -f "$BE_NAME" "$FE_NAME" "$PG_NAME" >/dev/null 2>&1 || true
docker network rm "$NET" >/dev/null 2>&1 || true
docker network create "$NET" >/dev/null
ok "Network $NET created"

# main.py calls init_db() at startup, so the backend needs a real database or it
# crashes on boot and the health check below would report a false failure.
if printf '%s\n' "${LAYERS[@]}" | grep -qx backend; then
  # Hostname `postgres` matches the POSTGRES_HOST default in services/database.py.
  docker run -d --name "$PG_NAME" --network "$NET" --network-alias postgres \
    -e POSTGRES_USER="$PG_USER" -e POSTGRES_PASSWORD="$PG_PASS" -e POSTGRES_DB="$PG_DB" \
    postgres:18 >/dev/null
  info "Waiting for Postgres..."
  for _ in $(seq 1 30); do
    if docker exec "$PG_NAME" pg_isready -U "$PG_USER" -d "$PG_DB" >/dev/null 2>&1; then
      ok "Postgres ready (db=$PG_DB user=$PG_USER — cluster identity, not compose's)"
      break
    fi
    sleep 1
  done

  docker run -d --name "$BE_NAME" --network "$NET" \
    ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} \
    -p "${BACKEND_PORT}:8080" \
    -e CORS_ORIGINS="https://${URL_FRONTEND}" \
    -e POSTGRES_HOST=postgres \
    -e POSTGRES_DB="$PG_DB" \
    -e POSTGRES_USER="$PG_USER" \
    -e POSTGRES_PASSWORD="$PG_PASS" \
    "gnn-view-backend:prebuild" >/dev/null
  ok "Backend started (CORS_ORIGINS=https://${URL_FRONTEND})"
fi

if printf '%s\n' "${LAYERS[@]}" | grep -qx frontend; then
  docker run -d --name "$FE_NAME" --network "$NET" \
    ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} \
    -p "${FRONTEND_PORT}:3000" \
    -e BACKEND_URL="${URL_BACKEND}" \
    "gnn-view-frontend:prebuild" >/dev/null
  ok "Frontend started (BACKEND_URL=${URL_BACKEND}, scheme-less as CI sets it)"
fi

# ── Health ───────────────────────────────────────────────────────────────────
# A container that dies here is exactly the CrashLoopBackOff the cluster would show.
probe() {
  local name="$1" url="$2" container="$3"
  local start=$SECONDS elapsed

  while :; do
    elapsed=$((SECONDS - start))

    if ! docker ps --format '{{.Names}}' | grep -qx "$container"; then
      fail "$name container exited after ${elapsed}s — would be CrashLoopBackOff in the cluster."
      printf "%s--- last 30 log lines ---%s\n" "$DIM" "$RESET"
      docker logs --tail 30 "$container" 2>&1 | sed 's/^/      /'
      return 1
    fi

    if curl -sf -o /dev/null "$url"; then
      if [ "$elapsed" -gt "$LIVENESS_BUDGET" ]; then
        fail "$name answered after ${elapsed}s — EXCEEDS the ${LIVENESS_BUDGET}s liveness budget; the cluster would kill it."
      elif [ "$elapsed" -gt "$READINESS_BUDGET" ]; then
        warn "$name answered after ${elapsed}s — past the ${READINESS_BUDGET}s readiness budget; rollouts will be flaky."
      else
        ok "$name healthy in ${elapsed}s (budget ${READINESS_BUDGET}s readiness / ${LIVENESS_BUDGET}s liveness)"
      fi
      return 0
    fi

    if [ "$elapsed" -gt $((LIVENESS_BUDGET * 2)) ]; then
      fail "$name never answered $url within $((LIVENESS_BUDGET * 2))s."
      printf "%s--- last 30 log lines ---%s\n" "$DIM" "$RESET"
      docker logs --tail 30 "$container" 2>&1 | sed 's/^/      /'
      return 1
    fi
    sleep 1
  done
}

step "Health probes"
[ -n "$PLATFORM" ] && [ "$(uname -m)" = "arm64" ] && \
  info "Timings under emulation are inflated — re-run with --native to judge real startup cost."

if printf '%s\n' "${LAYERS[@]}" | grep -qx backend; then
  probe "backend" "http://localhost:${BACKEND_PORT}/health" "$BE_NAME" || true
fi
if printf '%s\n' "${LAYERS[@]}" | grep -qx frontend; then
  probe "frontend" "http://localhost:${FRONTEND_PORT}/" "$FE_NAME" || true
fi

# ── Summary ──────────────────────────────────────────────────────────────────
step "Summary"

if [ "$FAILURES" -gt 0 ]; then
  printf "  %s%d failure(s), %d warning(s)%s — CI would likely fail. Fix before pushing.\n" \
    "$RED" "$FAILURES" "$WARNINGS" "$RESET"
  exit 1
fi

if [ "$WARNINGS" -gt 0 ]; then
  printf "  %s%d warning(s)%s — images build and boot, but review the notes above.\n" \
    "$YELLOW" "$WARNINGS" "$RESET"
else
  printf "  %sAll checks passed.%s Images build and boot as CI would produce them.\n" "$GREEN" "$RESET"
fi

cat <<EOF

${DIM}Not covered here — these need cluster access (kubectl -n gnn-view):
  rollout actually happened (both images are tagged :latest), OOMKills,
  PVC ReadWriteOnce mount conflicts, ingress/TLS, Harbor image pull.${RESET}
EOF
