import { defineConfig } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  testDir: "./tests",
  timeout: 5 * 60 * 1000, // 5 minutes — separation is slow
  expect: {
    timeout: 10_000,
  },
  fullyParallel: false, // Serial: GPU contention + shared IndexedDB
  workers: 1,
  retries: 0,
  reporter: "html",

  use: {
    baseURL: "http://localhost:4173",
    trace: "on-first-retry",
    video: "off",
  },

  projects: [
    {
      name: "chromium",
      use: {
        channel: "chrome",
        launchOptions: {
          args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
        },
      },
    },
  ],

  webServer: {
    command: "pnpm exec vite preview --port 4173",
    port: 4173,
    reuseExistingServer: !process.env.CI,
    cwd: path.resolve(__dirname, ".."),
  },
});
