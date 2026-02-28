import { test as base, expect, type Page } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";

/**
 * Test audio fixture: CC0 "There Ain't Nothin" by HoliznaCC0
 * https://freemusicarchive.org/music/holiznacc0/orphaned-media/there-aint-nothin
 * Downloaded by CI or `pnpm run e2e:setup`.
 */
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEST_AUDIO = path.resolve(__dirname, "test-audio.mp3");

/**
 * Page Object Model for the Demucs web app.
 */
export class DemucsApp {
  constructor(public readonly page: Page) {}

  // ── Navigation ──────────────────────────────────────────────────────────

  async goto() {
    await this.page.goto("/");
  }

  /** Wait for WASM + optional GPU warmup to complete. Tagline switches to "Source Separation". */
  async waitForWarmup() {
    await expect(this.page.locator(".tagline")).toHaveText("Source Separation", {
      timeout: 60_000,
    });
  }

  // ── Audio loading ───────────────────────────────────────────────────────

  /** Load the test audio fixture via the hidden file input. */
  async loadAudioFile(filePath: string = TEST_AUDIO) {
    const input = this.page.locator('.drop-zone input[type="file"]');
    await input.setInputFiles(filePath);
  }

  /** Wait for spectrogram canvas to be visible + run button ready. */
  async waitForSpectrogram() {
    await expect(this.page.locator(".spectrogram-canvas")).toBeVisible({
      timeout: 30_000,
    });
    await expect(this.page.locator(".run-btn")).toBeVisible({ timeout: 10_000 });
  }

  // ── Model selection ─────────────────────────────────────────────────────

  /** Click a model card by index (0 = htdemucs, 1 = htdemucs_6s, 2 = htdemucs_ft). */
  async selectModel(idx: number) {
    await this.page.locator(".model-card").nth(idx).click();
  }

  /** Wait for a model card's "Ready" badge to appear. */
  async waitForModelCached(idx: number) {
    await expect(
      this.page.locator(".model-card").nth(idx).locator(".model-card__cached"),
    ).toBeVisible({ timeout: 5 * 60_000 });
  }

  // ── Separation ──────────────────────────────────────────────────────────

  async clickRun() {
    await this.page.locator(".run-btn").click();
  }

  /** Wait for separation to complete (stem results visible). */
  async waitForSeparation() {
    await expect(this.page.locator(".stem-results")).toBeVisible({
      timeout: 5 * 60_000,
    });
  }

  // ── Helpers ─────────────────────────────────────────────────────────────

  /** Get the spectrogram wrapper bounding box. */
  async spectrogramBounds() {
    return this.page.locator(".spectrogram-wrap").boundingBox();
  }
}

/**
 * Extended test fixture that provides a `demucsApp` Page Object Model.
 */
export const test = base.extend<{ demucsApp: DemucsApp }>({
  demucsApp: async ({ page }, use) => {
    const app = new DemucsApp(page);
    await use(app);
  },
});

export { expect };
