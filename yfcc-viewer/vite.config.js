import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.BACKEND_API_URL ?? "http://127.0.0.1:8000";

  return {
    plugins: [react()],
    build: {
      outDir: "../src/yfcc_scope/dist/",
      emptyOutDir: true,
      sourcemap: false,
    },
    server: {
      proxy: {
        "/api": { target: target, changeOrigin: true },
        "/boxviewer": { target: target, changeOrigin: true },
        "/static": { target: target, changeOrigin: true },
      },
    },
  };
});
