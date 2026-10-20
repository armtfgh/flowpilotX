import { expect, test } from "@playwright/test";

test("real stages, gas setpoints, response history and full council transcript", async ({ page }, info) => {
  const run = process.env.FLOWPILOT_LIVE_RUN_ID;
  test.skip(!run, "Saved model run required");
  const api = await page.request.get(`/api/runs/${run}`);
  expect(api.ok()).toBe(true);
  const result = await api.json();
  const report = result.result_report;
  test.skip(!report.responses.some((r: any) => r.question_id === "Q-GAS-003"), "This gas-specific scenario requires a gas-intake fixture; use scientific-policy tests for thermal liquid designs");
  expect(report.issues).toEqual([]);
  const pageErrors: string[] = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  await page.goto(`/?run=${run}`);
  await expect(page.getByRole("heading", {name: "Stage-by-stage design"})).toBeVisible();
  const screenshot = async (name: string) => {
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
    await page.locator(".resultWorkspace").screenshot({path: info.outputPath(`${name}.png`), style: ".topbar { visibility: hidden !important; }"});
  };
  await screenshot("overview");
  await page.getByRole("button", {name: "Process summary", exact: true}).click();
  await expect(page.locator(".streamTable tbody tr")).toHaveCount(report.streams.length);
  for (const stream of report.streams) {
    const flow = String(Number(stream.flow_mL_min.toPrecision(6)));
    await expect(page.locator(".streamTable").getByText(flow, {exact: true})).toBeVisible();
  }
  expect(await page.locator(".resultBody").innerText()).not.toMatch(/sccm|in.channel/i);
  await screenshot("process-summary");
  const downloading = page.waitForEvent("download");
  await page.getByRole("button", {name: "CSV", exact: true}).click();
  await (await downloading).saveAs(info.outputPath("process-summary.csv"));
  await page.getByRole("button", {name: "Responses", exact: true}).click();
  await expect(page.locator(".responseRecord")).toHaveCount(report.responses.length);
  const history = page.locator(".responseRecord").filter({hasText: "Q-HIST-001"});
  await history.locator(":scope > summary").click();
  await expect(history.getByRole("heading", {name: "Answer · unavailable"})).toBeVisible();
  await expect(history.getByText("Explicitly unavailable", {exact: true})).toBeVisible();
  const gas = page.locator(".responseRecord").filter({hasText: "Q-GAS-003"});
  await gas.locator(":scope > summary").click();
  await expect(gas.getByText(/introduced at stage 2/)).toBeVisible();
  await screenshot("responses");
  await page.getByRole("button", {name: "Council", exact: true}).click();
  const records = (result.deliberation_log?.rounds || []).flat();
  await expect(page.locator(".agentRecord")).toHaveCount(records.length);
  const chemistry = page.locator(".agentRecord").filter({hasText: "Dr. Chemistry"}).first();
  await chemistry.locator(":scope > summary").click();
  const source = records.find((r: any) => r.agent === "DrChemistryV4");
  await expect(chemistry.locator(".recordText").first()).toHaveText(source.chain_of_thought);
  await screenshot("council");
  await page.getByRole("button", {name: "Engineering", exact: true}).click();
  await expect(page.getByText("1. Before council", {exact: true})).toBeVisible();
  await expect(page.getByText("2. Council-selected candidate", {exact: true})).toBeVisible();
  if (result.engineering_history?.before_council?.calculations?.steps?.length) {
    await page.getByText("Initial engineering calculator", {exact: true}).click();
    const steps = page.locator(".engineeringHistory > details").first().locator(".calculatorStep");
    await expect(steps).toHaveCount(result.engineering_history.before_council.calculations.steps.length);
    await steps.first().locator("summary").click();
    await expect(steps.first().locator(".recordText")).toContainText("Batch Conditions");
  }
  if (result.final_stage_engineering?.stages?.length) {
    const detail = page.locator(".finalEngineering details").first();
    await detail.locator("summary").click();
    await expect(detail.getByText("Reactor volume (mL)", {exact: true})).toBeVisible();
  }
  await screenshot("engineering");
  await page.getByRole("button", {name: "Process", exact: true}).click();
  const image = page.getByAltText("FlowPilot process topology");
  await expect.poll(() => image.evaluate((el: HTMLImageElement) => el.complete && el.naturalWidth > 100)).toBe(true);
  const svgResponse = await page.request.get(await image.getAttribute("src") || "");
  const svg = await svgResponse.text();
  expect(svg).toContain("mL/min at STP");
  expect(svg).not.toMatch(/sccm|\u2026|τch|in.channel/);
  expect(svg).toContain("data:image/png;base64,");
  await screenshot("topology");
  const width = await image.evaluate(el => el.getBoundingClientRect().width);
  await page.getByRole("button", {name: "Zoom in", exact: true}).click();
  expect(await image.evaluate(el => el.getBoundingClientRect().width)).toBeGreaterThan(width);
  await page.getByRole("button", {name: "Fit diagram", exact: true}).click();
  expect(pageErrors).toEqual([]);
});

test("single-stage overview and tables share exact final report, including multiple gas feeds", async ({page}, info) => {
  const run = process.env.FLOWPILOT_LIVE_RUN_ID;
  test.skip(!run, "Saved model run required as UI fixture base");
  const result = await (await page.request.get(`/api/runs/${run}`)).json();
  const stage = {...result.result_report.stages[1], number: 1, name: "Synthetic single-stage UI test",
    volume_mL: 10, liquid_flow_mL_min: 0.1, gas_flow_stp_mL_min: 0.9, residence_time_min: 10,
    temperature_C: 55, closure: true};
  result.result_report.stages = [stage];
  result.final_design.stages = [{stage_number: 1}];
  // Deliberately conflicting aggregate to detect an accidental legacy display path.
  result.final_design.parameters = {reactor_volume_mL: 999, flow_rate_mL_min: 999, residence_time_min: 999};
  result.result_report.streams = [
    {label: "A", phase: "liquid", introduction_stage: 1, contents: ["substrate"], flow_mL_min: 0.1, flow_basis: "liquid"},
    {label: "G1", phase: "gas", introduction_stage: 1, contents: ["O2"], flow_mL_min: 0.4, equiv: 2, flow_basis: "inlet/STP"},
    {label: "G2", phase: "gas", introduction_stage: 1, contents: ["N2"], flow_mL_min: 0.5, equiv: 0, flow_basis: "inlet/STP"},
  ];
  await page.route(`**/api/runs/${run}`, route => route.fulfill({json: result}));
  await page.goto(`/?run=${run}`);
  await expect(page.locator(".metricBand")).toContainText("0.9 mL/min");
  await expect(page.locator(".metricBand")).toContainText("10 min");
  await expect(page.locator(".metricBand")).not.toContainText("999");
  await expect(page.locator(".conditionTable")).toContainText("55");
  await page.locator(".resultWorkspace").screenshot({path: info.outputPath("single-stage-overview.png"), style: ".topbar {visibility:hidden!important}"});
  await page.getByRole("button", {name: "Process summary", exact: true}).click();
  await expect(page.locator(".processTable tbody tr")).toHaveCount(1);
  await expect(page.locator(".streamTable tbody tr")).toHaveCount(3);
  await expect(page.locator(".processTable")).toContainText("0.9");
});

test("council preserves additional stored narratives", async ({page}) => {
  const run = process.env.FLOWPILOT_LIVE_RUN_ID;
  test.skip(!run, "Saved model run required as UI fixture base");
  const result = await (await page.request.get(`/api/runs/${run}`)).json();
  result.deliberation_log = {rounds: [[{agent: "Audit fixture", chain_of_thought: "Primary stored assessment.", content: "Additional stored recommendation."}]]};
  await page.route(`**/api/runs/${run}`, route => route.fulfill({json: result}));
  await page.goto(`/?run=${run}`);
  await page.getByRole("button", {name: "Council", exact: true}).click();
  await page.locator(".agentRecord > summary").click();
  await expect(page.getByText("Primary stored assessment.", {exact: true})).toBeVisible();
  await page.locator(".agentRecord details > summary").filter({hasText: "content"}).click();
  await expect(page.getByText("Additional stored recommendation.", {exact: true})).toBeVisible();
});

test("rendered SVG captions are inside the canvas and do not overlap icons or other captions", async ({page}) => {
  const run = process.env.FLOWPILOT_LIVE_RUN_ID;
  test.skip(!run, "Saved model run required");
  const response = await page.request.get(`/api/runs/${run}/artifacts/process-svg`);
  expect(response.ok()).toBe(true);
  await page.setContent(await response.text());
  const issues = await page.evaluate(() => {
    const svg = document.querySelector("svg")!;
    const canvas = svg.getBoundingClientRect();
    const texts = [...svg.querySelectorAll("text")].map(el => ({text: el.textContent, box: el.getBoundingClientRect()}));
    const images = [...svg.querySelectorAll("image")].map(el => el.getBoundingClientRect());
    const errors: string[] = [];
    const overlaps = (a: DOMRect, b: DOMRect) => Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1;
    for (let i = 0; i < texts.length; i++) {
      const {text, box} = texts[i];
      if (box.left < canvas.left - 1 || box.right > canvas.right + 1 || box.top < canvas.top - 1 || box.bottom > canvas.bottom + 1) errors.push(`Outside canvas: ${text}`);
      if (images.some(img => overlaps(box, img))) errors.push(`Caption covers icon: ${text}`);
      for (const other of texts.slice(i + 1)) if (overlaps(box, other.box)) errors.push(`Captions overlap: ${text} / ${other.text}`);
    }
    return errors;
  });
  expect(issues).toEqual([]);
});
