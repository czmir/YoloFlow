import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiProxy = process.env.YOLOFLOW_API_PROXY || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": apiProxy,
      "/ws": { target: apiProxy, ws: true },
    },
  },
});
