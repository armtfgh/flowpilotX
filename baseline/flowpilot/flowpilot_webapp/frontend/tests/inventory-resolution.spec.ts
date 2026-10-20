import { expect, test } from "@playwright/test";

const failedRun = "20260907_094109_webapp";
const updatedId = "khu_laboratory_inventory_updated_20260807";

async function openFailedRun(page: import("@playwright/test").Page, mobile: boolean) {
  test.skip((await page.request.get(`/api/runs/${failedRun}`)).status() === 404, "Archived KHU regression fixture is not installed");
  await page.route("**/api/runs", (route) => route.fulfill({ json: { runs: [{
    run_id: failedRun, design_status: "inventory_confirmation_required", confidence: "NOT_ASSESSED"
  }] } }));
  await page.goto("/");
  if (mobile) await page.getByRole("button", { name: "Menu" }).click();
  await page.getByRole("button", { name: "Saved runs" }).click();
  await page.getByRole("button", { name: "Open saved design" }).click();
  await expect(page.getByText("INV-PRESSURE-001", { exact: true })).toBeVisible();
}

async function noOverflow(page: import("@playwright/test").Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
}

test("archived KHU gap is actionable and changing profile rebinds the package", async ({ page }, testInfo) => {
  await openFailedRun(page, testInfo.project.name === "mobile");
  await expect(page.getByRole("heading", { name: "Equipment review required", exact: true })).toBeVisible();
  await expect(page.getByText("-- min", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Run with resolved inventory" })).toBeDisabled();
  await noOverflow(page);
  await page.screenshot({ path: testInfo.outputPath("khu-missing-bpr.png"), fullPage: true });
  await page.getByRole("button", { name: "Process", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Requirements topology", exact: true })).toBeVisible();
  await page.getByLabel("Inventory profile", { exact: true }).selectOption(updatedId);
  await expect(page.getByText("INV-PRESSURE-001", { exact: true })).toHaveCount(0);
  // This old archive predates the mandatory gas-introduction confirmation.
  // Resolving equipment must not bypass that newer chemistry input gate.
  await expect(page.getByRole("button", { name: "Run with resolved inventory" })).toBeDisabled();
  await expect(page.getByText("Precheck passed", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Return to intake", exact: true }).click();
  const gasStage = page.locator(".question").filter({ hasText: "Q-GAS-003" });
  await gasStage.locator("textarea").fill("Stage 2, aerobic oxidation.");
  await page.locator(".intakePanel .toggle input").uncheck({ force: true });
  await page.getByRole("button", { name: "Save answers", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Design input is frozen" })).toBeVisible();
  await page.getByLabel("Upstream chemistry model").selectOption("qwen3.6-27b");
  await page.getByLabel("Downstream and council model").selectOption("qwen3.6-27b");
  await expect(page.getByRole("button", { name: /Run FlowPilot design/ })).toBeEnabled();
  await noOverflow(page);
  await page.screenshot({ path: testInfo.outputPath("khu-updated-profile-ready.png"), fullPage: true });
  // Protocol edits must invalidate the frozen package in the browser too.
  await page.getByLabel("Initial batch protocol").fill("Different oxidation protocol.");
  await expect(page.getByRole("button", { name: /Run FlowPilot design/ })).toBeDisabled();
});

test("real saved design displays numerical stages and saved icon topology", async ({ page }, testInfo) => {
  const runId = process.env.FLOWPILOT_LIVE_RUN_ID;
  test.skip(!runId, "Set FLOWPILOT_LIVE_RUN_ID to check a saved real-model result or deterministic replay");
  const response = await page.request.get(`/api/runs/${runId}`);
  expect(response.ok()).toBe(true);
  const result = await response.json();
  expect(result.final_design.status).toBe("executable");
  await page.route("**/api/runs", (route) => route.fulfill({ json: { runs: [{ run_id: runId, final_design_status: "executable" }] } }));
  await page.goto("/");
  if (testInfo.project.name === "mobile") await page.getByRole("button", { name: "Menu" }).click();
  await page.getByRole("button", { name: "Saved runs" }).click();
  await page.getByRole("button", { name: "Open saved design" }).click();
  await expect(page.getByText("Executable screening design", { exact: true })).toBeVisible();
  await expect(page.locator(".kpiGrid .kpi").first().getByText("READY", { exact: true })).toBeVisible();
  const fmt = (value: number) => Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  if (result.instrument_manifest.some((item: any) => item.requires_pre_run_verification)) {
    await expect(page.getByText("Equipment verification required before laboratory use", { exact: true })).toBeVisible();
  }
  if (result.design_realization?.decisions?.some((item: any) => item.confirmation_required)) {
    await expect(page.getByText("Chemist review required before laboratory use", { exact: true })).toBeVisible();
  }
  if (result.final_design.stages.length > 1) {
    await expect(page.locator(".processTable tbody tr")).toHaveCount(result.final_design.stages.length);
    for (const stage of result.result_report.stages) {
      const mobile = testInfo.project.name === "mobile";
      const row = page.locator(`${mobile ? ".stageOverviewCompact section" : ".processTable tr"}[data-stage="${stage.number}"]`);
      await expect(row.getByText(String(Number(stage.residence_time_min.toPrecision(6))) + (mobile ? " min" : ""), {exact: true})).toBeVisible();
    }
  } else await expect(page.locator(".metricBand").getByText(`${fmt(result.final_design.parameters.residence_time_min)} min`, { exact: true })).toBeVisible();
  await page.evaluate(() => window.scrollTo(0, 0));
  await noOverflow(page);
  await page.screenshot({ path: testInfo.outputPath("saved-design-overview.png"), fullPage: true });
  await page.getByRole("button", { name: "Process", exact: true }).click();
  const image = page.getByAltText("FlowPilot process topology");
  await expect(image).toBeVisible();
  await expect.poll(() => image.evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 100)).toBe(true);
  expect(await image.evaluate((img) => img.getBoundingClientRect().right <= document.documentElement.clientWidth)).toBe(true);
  await expect(page.locator(".processNode")).toHaveCount(0);
  await noOverflow(page);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: testInfo.outputPath("saved-design-topology.png"), fullPage: true });
  await page.getByRole("button", { name: "Engineering", exact: true }).click();
  for (const stage of result.final_design.stages) {
    await expect(page.locator(".finalEngineering").getByText(String(Number((stage.residence_time_inlet_min ?? stage.residence_time_min).toPrecision(6))), { exact: true }).first()).toBeVisible();
  }
  await noOverflow(page);
  await page.getByRole("button", { name: "Chemistry", exact: true }).click();
  for (const step of result.final_design.operating_procedure || []) {
    await expect(page.locator(".procedure").getByText(step.instruction.replace(/\bsccm\b/gi, "mL/min at STP"), { exact: true })).toBeVisible();
  }
  await noOverflow(page);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: testInfo.outputPath("saved-design-chemistry.png"), fullPage: true });
});

test("confirmation form handles available and unavailable without restarting intake", async ({ page }, testInfo) => {
  await openFailedRun(page, testInfo.project.name === "mobile");
  let submitted: any;
  // Persistence is tested against the real API in pytest; do not edit the user's KHU profile in this browser test.
  await page.route("**/api/inventory/resolve", async (route) => {
    const request = route.request().postDataJSON();
    submitted = request;
    const profile = structuredClone(request.inventory_profile);
    profile.version += 1;
    profile.lab_inventory.capability_status = { pressure_controllers: request.status };
    profile.lab_inventory.pressure_controllers = request.status === "available" ? [{...request.equipment, type: "BPR",
      setpoints_bar: request.equipment.setpoints_bar.split(",").map(Number),
      max_pressure_bar: Number(request.equipment.max_pressure_bar), quantity: Number(request.equipment.quantity) }] : [];
    const review = await page.request.post("/api/inventory/review", { data: {
      intake_package: request.intake_package, inventory_profile: profile, chemistry_plan: request.chemistry_plan
    } });
    expect(review.ok()).toBe(true);
    await route.fulfill({ json: { ...await review.json(), profile, saved_new_version: true } });
  });
  await page.getByLabel("Equipment availability", { exact: true }).selectOption("unavailable");
  await page.getByRole("button", { name: "Save inventory confirmation" }).click();
  await expect(page.getByText("Required equipment unavailable", { exact: true })).toBeVisible();
  expect(submitted.status).toBe("unavailable");
  await expect(page.getByRole("button", { name: "Run with resolved inventory" })).toBeDisabled();
  await page.getByText("Back-pressure regulator: marked unavailable. Revise confirmation", { exact: true }).click();
  await page.getByLabel("Equipment availability", { exact: true }).selectOption("available");
  await page.getByLabel("Equipment ID", { exact: true }).fill("browser-test-bpr");
  await page.getByLabel("Equipment name", { exact: true }).fill("Test BPR");
  await page.getByLabel("Available setpoints (bar gauge, comma-separated)", { exact: true }).fill("3");
  await page.getByLabel("Maximum pressure (bar)", { exact: true }).fill("8");
  await page.getByRole("button", { name: "Save inventory confirmation" }).click();
  await expect(page.getByText("Precheck passed", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Run with resolved inventory" })).toBeDisabled();
  expect(submitted.equipment.equipment_id).toBe("browser-test-bpr");
  await noOverflow(page);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: testInfo.outputPath("khu-confirmed-bpr-ready.png"), fullPage: true });
});
