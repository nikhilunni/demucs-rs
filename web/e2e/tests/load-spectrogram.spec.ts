import { test, expect } from "../fixtures/app";
import { isCanvasNonBlank } from "../helpers/validation";

test.describe("Load audio + spectrogram @smoke", () => {
  test("renders spectrogram and shows run button after loading audio", async ({
    demucsApp,
  }) => {
    await demucsApp.goto();
    await demucsApp.waitForWarmup();

    // Load test audio via file input
    await demucsApp.loadAudioFile();

    // Wait for phase transition to "ready"
    await demucsApp.waitForSpectrogram();

    // Verify spectrogram canvas is not blank
    const nonBlank = await isCanvasNonBlank(
      demucsApp.page,
      ".spectrogram-canvas",
    );
    expect(nonBlank).toBe(true);

    // Run button should be visible (disabled until a model is cached, which is fine for smoke)
    await expect(demucsApp.page.locator(".run-btn")).toBeVisible();
  });
});
