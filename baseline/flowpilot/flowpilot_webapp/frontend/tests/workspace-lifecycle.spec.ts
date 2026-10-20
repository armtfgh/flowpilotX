import { expect, test } from "@playwright/test";

test("draft answers and selected models survive a page reload", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Initial batch protocol").fill("Oxidation of compound A to B in ethanol at 40 C for 2 hours.");
  await page.locator(".intakePanel .toggle input").uncheck({ force: true });
  await page.getByRole("button", { name: "Analyze intake" }).click();
  const objective = page.locator(".question").filter({ hasText: "Q-OBJ-001" }).locator("textarea");
  await objective.fill("Preserved draft objective.");
  await page.getByLabel("Upstream chemistry model").selectOption("qwen3.6-27b");
  await page.reload();
  await expect(objective).toHaveValue("Preserved draft objective.");
  await expect(page.getByLabel("Upstream chemistry model")).toHaveValue("qwen3.6-27b");
  await expect(page.getByLabel("Initial batch protocol")).toContainText("Oxidation of compound A");
});

test("reloading an active job resumes polling without submitting another design", async ({ page }, testInfo) => {
  const archive = await page.request.get("/api/runs/20260907_094109_webapp");
  test.skip(archive.status() === 404, "KHU regression fixture is not installed");
  const result = await archive.json();
  let complete = false;
  let submissions = 0;
  await page.route("**/api/design/jobs", async (route) => {
    if (route.request().method() === "POST") submissions++;
    await route.continue();
  });
  await page.route("**/api/design/jobs/lifecycle-test", (route) => route.fulfill({ json: {
    job_id: "lifecycle-test", status: complete ? "completed" : "running", progress: .58,
    phase: complete ? "Inventory confirmation required" : "Council review", messages: [],
    result: complete ? result : null
  } }));
  await page.goto("/?job=lifecycle-test");
  await expect(page.getByRole("heading", { name: "Council review", exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("heading", { name: "Council review", exact: true })).toBeVisible();
  complete = true;
  await expect(page.getByText("INV-PRESSURE-001", { exact: true })).toBeVisible();
  await expect(page.getByText("Job lifecycle-test", { exact: true })).toBeVisible();
  await expect(page.getByText("-- min", { exact: true })).toHaveCount(0);
  expect(submissions).toBe(0);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: testInfo.outputPath("resumed-job-recovery.png"), fullPage: true });
});

test("an outdated workspace blocks new submissions and reload preserves input", async ({ page }) => {
  const current = await (await page.request.get("/api/runtime")).json();
  let stale = true;
  await page.route("**/api/runtime", (route) => route.fulfill({ json: {
    ...current, frontend_build_id: stale ? "a-different-build" : current.frontend_build_id
  } }));
  await page.goto("/");
  await expect(page.getByText("Workspace update available.", { exact: false })).toBeVisible();
  await page.getByLabel("Initial batch protocol").fill("Retain this protocol through an update.");
  await expect(page.getByRole("button", { name: "Analyze intake" })).toBeDisabled();
  stale = false;
  await page.getByRole("button", { name: "Reload workspace" }).click();
  await expect(page.getByLabel("Initial batch protocol")).toHaveValue("Retain this protocol through an update.");
  await expect(page.getByRole("button", { name: "Analyze intake" })).toBeEnabled();
  await expect(page.getByText("Workspace update available.", { exact: false })).toHaveCount(0);
});

test("the run list includes a job before it has an autosaved final result", async ({ page }, testInfo) => {
  await page.route("**/api/runs", (route) => route.fulfill({ json: { runs: [] } }));
  const job = { job_id: "active-list-test", status: "running", progress: .58, phase: "Council review", messages: [] };
  await page.route("**/api/design/jobs", (route) => route.fulfill({ json: { jobs: [job] } }));
  await page.route("**/api/design/jobs/active-list-test", (route) => route.fulfill({ json: job }));
  await page.goto("/");
  if (testInfo.project.name === "mobile") await page.getByRole("button", { name: "Menu" }).click();
  await page.getByRole("button", { name: "Saved runs" }).click();
  await expect(page.getByRole("heading", { name: "In-progress designs" })).toBeVisible();
  await page.getByRole("button", { name: "Open current job" }).click();
  await expect(page.getByRole("heading", { name: "Council review", exact: true })).toBeVisible();
  await expect(page).toHaveURL(/job=active-list-test/);
  await expect(page.getByRole("heading", { name: "Standardized intake", exact: true })).toHaveCount(0);
});
