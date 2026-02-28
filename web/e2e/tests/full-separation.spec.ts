import { test, expect } from "../fixtures/app";
import { isCanvasNonBlank, computeStemSumSDR } from "../helpers/validation";

test.describe("Full separation E2E", () => {
  test("separates audio into 4 stems with correct SDR", async ({
    demucsApp,
  }) => {
    await demucsApp.goto();
    await demucsApp.waitForWarmup();

    // Load audio
    await demucsApp.loadAudioFile();
    await demucsApp.waitForSpectrogram();

    // Select model (assumes htdemucs is already cached from model-download test)
    await demucsApp.selectModel(0);
    await demucsApp.waitForModelCached(0);

    // Run separation
    await demucsApp.clickRun();
    await demucsApp.waitForSeparation();

    // Verify 4 stem rows with correct names
    const stemNames = demucsApp.page.locator(".stem-row__name");
    await expect(stemNames).toHaveCount(4);

    const names = await stemNames.allTextContents();
    expect(names).toEqual(
      expect.arrayContaining(["drums", "bass", "other", "vocals"]),
    );

    // Verify each stem spectrogram is non-blank
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

    // Validate stem sum quality via __e2e data
    const sdr = await computeStemSumSDR(demucsApp.page);
    expect(sdr).not.toBeNull();
    if (sdr) {
      console.log(
        `Stem sum SDR — left: ${sdr.sdrLeft.toFixed(1)} dB, right: ${sdr.sdrRight.toFixed(1)} dB`,
      );
      expect(sdr.sdrLeft).toBeGreaterThan(20);
      expect(sdr.sdrRight).toBeGreaterThan(20);
    }
  });
});
