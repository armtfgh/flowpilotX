import { expect, test } from "@playwright/test";

test("private policy is explicit, persists, and sends twelve candidates", async ({page}) => {
  const response = await page.request.get("/api/runs/20260907_145525_webapp");
  test.skip(!response.ok(), "Saved DPDTC intake needed");
  const source = await response.json();
  await page.goto("/");
  await page.evaluate((intake) => sessionStorage.setItem("flowpilot_workspace_v1", JSON.stringify({
    protocol: intake.raw_protocol, intakePackage: intake, selectedProfile: intake.inventory_profile_snapshot,
    upstreamModelId: "claude-opus-4-6", downstreamModelId: "claude-sonnet-4-6", designPolicy: "legacy"
  })), source.intake_package);
  await page.reload();
  await expect(page.getByLabel("Design policy")).toHaveValue("legacy");
  await page.getByLabel("Design policy").selectOption("scientific_v2");
  await page.reload();
  await expect(page.getByLabel("Design policy")).toHaveValue("scientific_v2");
  await page.getByLabel("Screening priority").selectOption("yield_priority");
  await page.reload();
  await expect(page.getByLabel("Screening priority")).toHaveValue("yield_priority");
  let sent: any;
  await page.route("**/api/design/jobs", async route => {
    if (route.request().method() !== "POST") return route.continue();
    sent = route.request().postDataJSON();
    await route.fulfill({status: 400, json: {detail: "Submission intercepted by browser test; no model call."}});
  });
  await page.getByRole("button", {name: /Run FlowPilot design/}).click();
  await expect.poll(() => sent?.runtime_options).toEqual({design_policy: "scientific_v2", candidate_budget: 12});
  expect(sent.intake_package.raw_protocol).toBe(source.intake_package.raw_protocol);
  expect(sent.intake_package.screening_priority).toBe("yield_priority");
});

test("scientific result shows full review coverage, final stages, and icon topology", async ({page}, info) => {
  const run = process.env.FLOWPILOT_SCIENTIFIC_RUN;
  test.skip(!run, "Completed private scientific run required");
  const result = await (await page.request.get(`/api/runs/${run}`)).json();
  expect(result.final_design.status).toBe("executable");
  expect(result.scientific_assessment.selected_design_preserved).toBe(true);
  const errors: string[] = [];
  page.on("pageerror", err => errors.push(err.message));
  await page.goto(`/?run=${run}`);
  await expect(page.getByRole("heading", {name: "Scientific assessment"})).toBeVisible();
  await expect.poll(() => page.evaluate(() => JSON.parse(sessionStorage.getItem("flowpilot_workspace_v1") || "{}").designPolicy)).toBe("scientific_v2");
  await expect(page.getByText("Experimental screen, not a validated high-yield process.", {exact:false})).toBeVisible();
  if (result.scientific_assessment.source_context?.gas_delivery?.requires_chemist_confirmation) {
    await expect(page.locator(".gasChangeWarning")).toBeVisible();
    await expect(page.locator(".gasChangeWarning")).toContainText("Unapproved gas-feed change");
  }
  if (result.scientific_assessment.objective_policy) {
    await expect(page.getByRole("heading", {name: "Objective and selection"})).toBeVisible();
    await page.getByText("Why alternatives were not selected", {exact:true}).click();
    await expect(page.locator(".objectiveAssessment .auditBody p")).toHaveCount(result.scientific_assessment.chief.alternatives.length);
  }
  for (const warning of await page.locator(".temperatureDeviation").all()) {
    const bounds = await warning.boundingBox();
    const text = await warning.locator("span").boundingBox();
    expect(text!.width).toBeGreaterThan(bounds!.width * 0.65);
    expect(bounds!.height).toBeLessThan(info.project.name === "mobile" ? 300 : 130);
  }
  await page.getByText("12-candidate screen and reviewer coverage", {exact:true}).click();
  await expect(page.locator(".scientificAssessment tbody tr")).toHaveCount(12);
  await expect(page.locator(".scientificAssessment tbody tr").filter({hasText:"4 / 4"})).toHaveCount(12);
  const screenshot = async (name: string) => {
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
    await page.locator(".resultWorkspace").screenshot({path: info.outputPath(`${name}.png`), style: ".topbar {visibility:hidden!important}"});
  };
  await screenshot("scientific-overview");
  await page.getByRole("button", {name:"Process summary", exact:true}).click();
  await expect(page.locator(".processTable tbody tr")).toHaveCount(2);
  await expect(page.locator(".streamTable tbody tr")).toHaveCount(2);
  await screenshot("scientific-summary");
  if (result.scientific_assessment.answer_effects) {
    await page.getByRole("button", {name:"Responses", exact:true}).click();
    const objective = page.locator(".responseRecord").filter({has: page.locator("summary code", {hasText:"Q-OBJ-001"})});
    await objective.locator("summary").first().click();
    await expect(objective).toContainText("Policy binding:");
    await expect(objective).toContainText(result.scientific_assessment.objective_policy.priority);
    await screenshot("scientific-responses");
  }
  await page.getByRole("button", {name:"Council", exact:true}).click();
  await expect(page.locator(".agentRecord")).toHaveCount(6);
  await page.locator(".agentRecord > summary").first().click();
  await expect(page.locator(".agentRecord .recordText").first()).toContainText("candidate_id");
  await screenshot("scientific-council");
  await page.getByRole("button", {name:"Process", exact:true}).click();
  if (result.scientific_assessment.source_context?.gas_delivery?.requires_chemist_confirmation) {
    await expect(page.locator(".gasChangeWarning")).toBeVisible();
  }
  const image = page.getByAltText("FlowPilot process topology");
  await expect.poll(() => image.evaluate((el: HTMLImageElement) => el.complete && el.naturalWidth > 100)).toBe(true);
  const svg = await (await page.request.get(await image.getAttribute("src") || "")).text();
  expect(svg).toContain("data:image/png;base64,");
  expect(svg).not.toMatch(/\u2026|sccm|in.channel/);
  await screenshot("scientific-topology");
  expect(errors).toEqual([]);
});
