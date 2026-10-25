import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    watch: {
      ignored: ["**/.venv/**", "**/ad-studio/script-brain/**", "**/ad-studio/script-brain/data/**"],
    },
  },
});
