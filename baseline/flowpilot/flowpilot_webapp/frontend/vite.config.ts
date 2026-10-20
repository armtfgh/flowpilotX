import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

const buildId = createHash("sha256")
  .update(readFileSync(new URL("./src/main.tsx", import.meta.url)))
  .update(readFileSync(new URL("./src/result_views.tsx", import.meta.url)))
  .update(readFileSync(new URL("./src/styles.css", import.meta.url)))
  .digest("hex").slice(0, 12);

export default defineConfig({
  define: { __FLOWPILOT_UI_BUILD__: JSON.stringify(buildId) },
  plugins: [react(), {
    name: "flowpilot-build-identity",
    generateBundle() {
      this.emitFile({ type: "asset", fileName: "build.json", source: JSON.stringify({ frontend_build_id: buildId }) });
    }
  }],
  server: {
    port: 5174,
    proxy: { "/api": "http://127.0.0.1:8765" }
  }
});
