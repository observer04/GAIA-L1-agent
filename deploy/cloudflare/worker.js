export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Only handle strict /gaia_agent path. Let other routes pass through untouched.
    if (!url.pathname.startsWith("/gaia_agent")) {
      return fetch(request);
    }

    const origin = (env.AZURE_ORIGIN || "").trim();
    if (!origin) {
      return new Response("Missing AZURE_ORIGIN", { status: 500 });
    }

    const upstream = new URL(url.toString());
    upstream.protocol = "https:";
    upstream.hostname = origin;

    const upstreamRequest = new Request(upstream.toString(), request);
    const response = await fetch(upstreamRequest, {
      cf: {
        // Cache static assets under /gaia_agent/assets/*
        cacheEverything: upstream.pathname.startsWith("/gaia_agent/assets/"),
      },
    });

    const headers = new Headers(response.headers);

    // Never cache API responses.
    if (upstream.pathname.startsWith("/gaia_agent/api/")) {
      headers.set("Cache-Control", "no-store");
    }

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  },
};
