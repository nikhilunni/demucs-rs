import { test, expect } from "../fixtures/app";
import { isCanvasNonBlank, computeStemSumSDR } from "../helpers/validation";

test.describe("Clip selection + subsection separation", () => {
  test("separates a subsection of the audio", async ({ demucsApp }) => {
    await demucsApp.goto();
    await demucsApp.waitForWarmup();

    // Load audio
    await demucsApp.loadAudioFile();
    await demucsApp.waitForSpectrogram();

    // Get spectrogram bounds for mouse interaction
    const bounds = await demucsApp.spectrogramBounds();
    expect(bounds).not.toBeNull();
    if (!bounds) return;

    // Drag from 25% to 75% of the spectrogram width to create a clip selection
    const startX = bounds.x + bounds.width * 0.25;
    const endX = bounds.x + bounds.width * 0.75;
    const centerY = bounds.y + bounds.height / 2;

    await demucsApp.page.mouse.move(startX, centerY);
    await demucsApp.page.mouse.down();
    await demucsApp.page.mouse.move(endX, centerY, { steps: 10 });
    await demucsApp.page.mouse.up();

    // Verify clip selection is visible (clip overlay elements)
    await expect(demucsApp.page.locator(".clip-selection")).toBeVisible({
      timeout: 5_000,
    });

    // Run button text should include time range
    const runBtn = demucsApp.page.locator(".run-btn");
    await expect(runBtn).toContainText("Run separation (");

    // Select model and run separation on the clip
    await demucsApp.selectModel(0);
    await demucsApp.waitForModelCached(0);
    await demucsApp.clickRun();
    await demucsApp.waitForSeparation();

    // Verify stems are present
    const stemNames = demucsApp.page.locator(".stem-row__name");
    await expect(stemNames).toHaveCount(4);

    // Verify stem length is less than original (subsection)
    const stemInfo = await demucsApp.page.evaluate(() => {
      const e2e = (window as any).__e2e;
      if (!e2e) return null;
      return {
        stemLength: e2e.stems[0]?.left?.length ?? 0,
        originalLength: e2e.originalLeft?.length ?? 0,
      };
    });
    expect(stemInfo).not.toBeNull();
    if (stemInfo) {
      expect(stemInfo.stemLength).toBeLessThan(stemInfo.originalLength);
      expect(stemInfo.stemLength).toBeGreaterThan(0);
    }

    // Verify stem spectrograms render
    const stemSpectrograms = demucsApp.page.locator(
      ".stem-row__spectrogram canvas",
    );
    const count = await stemSpectrograms.count();
    expect(count).toBe(4);

    for (let i = 0; i < count; i++) {
      const nonBlank = await isCanvasNonBlank(
        demucsApp.page,
        `.stem-row:nth-child(${i + 1}) .stem-row__spectrogram canvas`,
      );
      expect(nonBlank).toBe(true);
    }

    // Validate stem sum quality
    const sdr = await computeStemSumSDR(demucsApp.page);
    expect(sdr).not.toBeNull();
    if (sdr) {
      console.log(
        `Clip SDR — left: ${sdr.sdrLeft.toFixed(1)} dB, right: ${sdr.sdrRight.toFixed(1)} dB`,
      );
      expect(sdr.sdrLeft).toBeGreaterThan(20);
      expect(sdr.sdrRight).toBeGreaterThan(20);
    }
  });
});
