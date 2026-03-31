# Operations runbook and rollback

This runbook covers rollout and rollback for the GAIA app served at `/gaia_agent`.

## Pre-rollout checks

- Ensure backend/frontend CI workflows are green.
- Verify container image exists in ACR for target commit.
- Verify runtime env vars are configured in Container Apps.
- Ensure Cloudflare route worker is deployed (`deploy-cloudflare-worker.yml`) when `/gaia_agent` routing has changed.
- Run smoke checks against the current production URL.

## Rollout (Azure Container Apps)

1. Trigger `.github/workflows/deploy-gaia-agent.yml`.
2. Provide `baseUrl` (or keep secret `GAIA_DEPLOY_BASE_URL` set) and set `skipChat=false` for full gate.
3. Optionally set `rollbackOnFail=true` for automatic image rollback when smoke fails.
4. If smoke is green, keep the revision active.

## Smoke validation (manual)

Run local script:

- `python scripts/rollout_smoke.py --base-url https://ommprakash.cloud/gaia_agent`

Optional preflight without chat call:

- `python scripts/rollout_smoke.py --base-url https://ommprakash.cloud/gaia_agent --skip-chat`

## Runtime guardrails

Recommended production env values:

- `WEB_BASE_PATH=/gaia_agent`
- `WEB_PUBLIC_MODE=true`
- `AGENT_ALLOW_UNSAFE_TOOLS=false`
- `WEB_RATE_LIMIT_ENABLED=true`
- `WEB_RATE_LIMIT_REQUESTS_PER_MINUTE=30` (tune by traffic)
- `WEB_RATE_LIMIT_WINDOW_SECONDS=60`
- `WEB_ENABLE_HSTS=true` (enable only when served through HTTPS)

## Rollback strategy

If smoke fails post-deploy:

1. Identify last known-good image tag (previous SHA).
2. Update Container App back to previous image.
3. Re-run smoke checks.
4. Keep Cloudflare route unchanged (path remains `/gaia_agent`).

If Azure has an incident and traffic must be diverted:

1. Keep DNS at Cloudflare.
2. Temporarily change Cloudflare origin rule for `/gaia_agent*` to fallback origin.
3. Run smoke checks against public URL.
4. Restore primary origin once healthy.

## Incident notes template

- Incident start time:
- User impact:
- Failed checks:
- Immediate mitigation:
- Root cause:
- Follow-up fix:
