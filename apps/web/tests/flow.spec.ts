import { test, expect } from "@playwright/test";
import path from "node:path";
const cat = path.resolve("../../data/demo/cat.png");
test("upload, preview, train, progressive ink, pause, export, replace and reset", async ({
  page,
}) => {
  const failures: string[] = [];
  page.on("pageerror", (e) => failures.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: /Send the fly/ }),
  ).toBeVisible();
  await page.getByLabel("Upload picture").setInputFiles(cat);
  await expect(page.getByAltText("Original uploaded image")).toBeVisible();
  await expect(page.getByAltText("Simplified sketch target")).toBeVisible();
  await page.getByLabel("Generations", { exact: true }).fill("1");
  const started = page.waitForResponse((r) => r.url().endsWith("/api/start"));
  await page.getByRole("button", { name: "Send to art school" }).click();
  const run = await (await started).json();
  await expect(page.getByText(/256 recorded movements/)).toBeVisible({
    timeout: 60000,
  });
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Resume", exact: true }),
  ).toBeVisible();
  const progress = await page.locator(".caption").first().textContent();
  await page.waitForTimeout(300);
  expect(await page.locator(".caption").first().textContent()).toEqual(
    progress,
  );
  await page.getByRole("button", { name: "Resume", exact: true }).click();
  await page.getByLabel("Replay speed").selectOption("8");
  await expect
    .poll(async () => page.locator(".caption").first().textContent())
    .toContain("256 / 256");
  const hasInk = await page
    .getByLabel("Live fly drawing")
    .evaluate((c: HTMLCanvasElement) => {
      const data = c
        .getContext("2d")!
        .getImageData(0, 0, c.width, c.height).data;
      for (let i = 0; i < data.length; i += 4) if (data[i] < 80) return true;
      return false;
    });
  expect(hasInk).toBeTruthy();
  const exported = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export PNG" }).click();
  expect((await exported).suggestedFilename()).toBe("fly-gogh-drawing.png");
  await expect(page.locator(".canvas-header")).toContainText("DONE", {
    timeout: 60000,
  });
  await expect
    .poll(async () => page.locator(".caption").first().textContent())
    .toContain("256 / 256");
  await page.screenshot({ path: "../../runs/ui-desktop.png", fullPage: true });
  await page.getByRole("button", { name: "leaf", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Send to art school" }),
  ).toBeEnabled();
  await expect(page.getByRole("button", { name: "Export PNG" })).toBeDisabled();
  expect((await page.request.get(`/api/runs/${run.run_id}`)).status()).toBe(
    404,
  );
  await page.getByRole("button", { name: "Send to art school" }).click();
  await expect(
    page.getByRole("button", { name: "Reset", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Send to art school" }),
  ).toBeEnabled();
  expect(failures).toEqual([]);
});
test("bad upload, blank target, detail changes and mobile layout", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page
    .getByLabel("Upload picture")
    .setInputFiles({
      name: "bad.png",
      mimeType: "image/png",
      buffer: Buffer.from("bad"),
    });
  await expect(page.getByRole("alert")).toContainText("Cannot decode");
  await page.getByRole("button", { name: "Dismiss error" }).click();
  await page
    .getByLabel("Upload picture")
    .setInputFiles(path.resolve("../../data/demo/blank.png"));
  await expect(
    page.getByText("Target is nearly blank.", { exact: false }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Send to art school" }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "cat", exact: true }).click();
  await page.getByLabel("Target pixels").selectOption("64");
  await page.getByRole("button", { name: "Apply detail" }).click();
  await expect(page.getByAltText("Simplified sketch target")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({ path: "../../runs/ui-mobile.png", fullPage: true });
});
