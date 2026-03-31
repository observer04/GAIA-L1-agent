# Deployment scaffolds for `/gaia_agent`

This folder contains practical scaffolds for running the GAIA FastAPI + React app behind Cloudflare with Azure as origin.

## 1) Azure origin

Use the provided GitHub Actions workflow:

- `.github/workflows/deploy-gaia-agent.yml` (recommended: deploy + smoke + optional rollback)
- `.github/workflows/deploy-azure-containerapp.yml`
- `.github/workflows/deploy-cloudflare-worker.yml` (publish strict `/gaia_agent` route worker)

Optional Modal deploy entrypoint:

- `modal_app.py` (`modal deploy modal_app.py`)

Required repository secrets:

- `AZURE_CREDENTIALS`
- `AZURE_CONTAINER_REGISTRY`
- `AZURE_RESOURCE_GROUP`
- `AZURE_CONTAINER_APP`

Cloudflare workflow secrets:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`
- optional `BACKEND_ORIGIN_HOST` (preferred generic host)
- optional `AZURE_ORIGIN_HOST` (legacy fallback)

Container runtime env expectations:

- `WEB_BASE_PATH=/gaia_agent`
- `WEB_STATIC_DIR=/app/frontend/dist`
- `WEB_PUBLIC_MODE=true`
- `AGENT_ALLOW_UNSAFE_TOOLS=false`
- `WEB_RATE_LIMIT_ENABLED=true`
- `WEB_RATE_LIMIT_REQUESTS_PER_MINUTE=30`
- `WEB_RATE_LIMIT_WINDOW_SECONDS=60`
- `WEB_ENABLE_HSTS=true` (only when HTTPS is enforced end-to-end)

## 2) Cloudflare edge routing

Use Cloudflare to keep the public URL stable at:

- `https://ommprakash.cloud/gaia_agent`

Recommended edge setup:

1. Create/proxy DNS record for your Azure origin hostname.
2. Create an Origin Rule so requests matching `/gaia_agent*` are forwarded to Azure origin.
3. Add a Cache Rule to bypass cache for `/gaia_agent/api*`.
4. Optionally enable cache for static assets under `/gaia_agent/assets/*`.

See `deploy/cloudflare/path-routing-checklist.md` for an operator checklist.
If you prefer Worker-based path routing, start from:

- `deploy/cloudflare/worker.example.js`
- `deploy/cloudflare/wrangler.example.toml`
- production worker entrypoint used by CI: `deploy/cloudflare/worker.js`

For Modal as backend origin, set `BACKEND_ORIGIN` to the deployed host (for example `<workspace>--gaia-agent-api-fastapi-app.modal.run`).

## 3) Rollout validation

- Local/ops smoke script: `scripts/rollout_smoke.py`
- GitHub Action smoke workflow: `.github/workflows/rollout-smoke.yml`
- Runbook + rollback guide: `deploy/operations-runbook.md`
