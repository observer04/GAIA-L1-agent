# Cloudflare path routing checklist

Goal: serve the app at `https://<your-domain>/gaia_agent` while Azure Container Apps remains the origin.

## Checklist

- [ ] DNS: create a proxied DNS record for the Azure origin host.
- [ ] Origin Rule: route requests where URI path starts with `/gaia_agent` to Azure origin.
- [ ] Cache Rule: bypass cache for `/gaia_agent/api*`.
- [ ] Cache Rule: allow cache (with sensible TTL) for `/gaia_agent/assets/*`.
- [ ] (Optional) WAF rule: throttle abusive traffic for `/gaia_agent/api/chat*`.
- [ ] Validate:
  - [ ] `GET /gaia_agent/api/health` returns `200`.
  - [ ] `GET /gaia_agent/` loads SPA shell.
  - [ ] Chat streaming works from browser.

## Notes

- Keep `WEB_BASE_PATH=/gaia_agent` in the Azure runtime to ensure docs, API endpoints, and SPA fallback stay prefix-correct.
- If your origin rewrites host headers, ensure the forwarded host still matches your app expectations.
- Worker template is available at `deploy/cloudflare/worker.example.js` with sample `wrangler` config at `deploy/cloudflare/wrangler.example.toml`.
- Automated deployment path is available via `.github/workflows/deploy-cloudflare-worker.yml`.
