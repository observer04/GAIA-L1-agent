import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ command }) => {
  const configuredBase = process.env.VITE_BASE_PATH;
  const basePath = configuredBase || (command === "build" ? "/gaia_agent/" : "/");

  return {
    plugins: [react()],
    base: basePath,
    server: {
      port: 5173,
      proxy: {
        "/api": "http://localhost:8000",
        "/gaia_agent/api": {
          target: "http://localhost:8000",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/gaia_agent/, ""),
        },
      },
    },
  };
});
