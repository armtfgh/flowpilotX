import { expect, test } from "@playwright/test";

test("saved backflow review remains visible with numerical results and topology", async ({page}, info) => {
  const run = process.env.FLOWPILOT_BACKFLOW_RUN_ID;
  test.skip(!run, "Requires an archived backflow-on wiring or live-model run");
  const response = await page.request.get(`/api/runs/${run}`);
  expect(response.ok()).toBe(true);
  const result = await response.json();
  expect(result.final_design.flow_operability.applicable).toBe(true);
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(`/?run=${run}`);
  await expect(page.getByText("Screening proposal: laboratory review required", {exact: true})).toBeVisible();
  await expect(page.getByRole("heading", {name: "Stage-by-stage design"})).toBeVisible();
  await expect(page.locator(".resultBanner")).toHaveClass(/review/);
  await expect(page.locator(".checkList .fail").filter({hasText: "Flow operability"})).toBeVisible();
  const screenshot = async (name: string) => {
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
    await page.locator(".resultWorkspace").screenshot({path: info.outputPath(`${name}.png`),
      style: ".topbar {visibility:hidden!important}"});
  };
  await screenshot("backflow-overview");
  await page.getByRole("button", {name: "Council", exact: true}).click();
  const review = page.locator("details.auditSection").filter({has: page.locator(":scope > summary", {hasText: "Flow operability"})}).first();
  await expect(review).toBeVisible();
  for (const finding of result.final_design.flow_operability.findings) {
    await expect(review.getByText(finding.message, {exact: true})).toBeVisible();
  }
  await expect(review).toContainText("Lower gas flow does not establish protection against backflow");
  await screenshot("backflow-council");
  await page.getByRole("button", {name: "Process summary", exact: true}).click();
  await expect(page.locator(".streamTable tbody tr")).toHaveCount(result.result_report.streams.length);
  const gas = result.result_report.streams.find((s: any) => s.phase === "gas");
  await expect(page.locator(".streamTable")).toContainText(String(Number(gas.flow_mL_min.toPrecision(6))));
  await screenshot("backflow-streams");
  await page.getByRole("button", {name: "Process", exact: true}).click();
  await expect(page.getByRole("heading", {name: "Proposed process topology", exact: true})).toBeVisible();
  const image = page.getByAltText("FlowPilot process topology");
  await expect.poll(() => image.evaluate((el: HTMLImageElement) => el.complete && el.naturalWidth > 100)).toBe(true);
  const svg = await (await page.request.get(await image.getAttribute("src") || "")).text();
  expect(svg.includes("data:image/png;base64,")).toBe(true);
  const labels = await page.evaluate(source => [...new DOMParser().parseFromString(source, "image/svg+xml")
    .querySelectorAll("text")].map(t => t.textContent).join("\n"), svg);
  expect(labels).toContain("mL/min at STP");
  if (result.final_design.flow_operability.revision_status === "proposed_gas_source_revision") {
    expect(labels).toContain("O2");
    expect(labels).toContain(String(gas.flow_mL_min));
    expect(labels).not.toContain("Protocol-authorized O2");
    expect(JSON.stringify(result.process_topology)).toContain("Council-proposed O2");
  }
  await screenshot("backflow-topology");
  expect(errors).toEqual([]);
});

test("feature-off saved run retains the original UI contract", async ({page}, info) => {
  const run = process.env.FLOWPILOT_BACKFLOW_OFF_RUN_ID;
  test.skip(!run, "Requires archived feature-off run");
  await page.goto(`/?run=${run}`);
  await expect(page.getByRole("heading", {name: "Stage-by-stage design"})).toBeVisible();
  await expect(page.getByText("Flow operability requires laboratory review", {exact: true})).toHaveCount(0);
  await page.locator(".resultWorkspace").screenshot({path: info.outputPath("feature-off-overview.png"),
    style: ".topbar {visibility:hidden!important}"});
});
