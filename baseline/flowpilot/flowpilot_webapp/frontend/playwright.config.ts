import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  expect: { timeout: 5_000 },
  use: {
    baseURL: process.env.FLOWPILOT_TEST_URL || "http://127.0.0.1:8510",
    trace: "retain-on-failure",
    launchOptions: process.env.FLOWPILOT_CHROME_PATH
      ? { executablePath: process.env.FLOWPILOT_CHROME_PATH }
      : undefined,
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1440, height: 960 } } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
  reporter: [["list"]],
});
