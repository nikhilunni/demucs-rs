import type { Page } from "@playwright/test";

/**
 * Check that a canvas element is not blank (i.e., not all pixels are the same color).
 * Samples a 10x10 block from the center of the canvas.
 */
export async function isCanvasNonBlank(
  page: Page,
  selector: string,
): Promise<boolean> {
  return page.evaluate((sel) => {
    const canvas = document.querySelector(sel) as HTMLCanvasElement | null;
    if (!canvas) return false;

    const ctx = canvas.getContext("2d");
    if (!ctx) return false;

    const cx = Math.floor(canvas.width / 2);
    const cy = Math.floor(canvas.height / 2);
    const size = 10;
    const x = Math.max(0, cx - size / 2);
    const y = Math.max(0, cy - size / 2);
    const data = ctx.getImageData(x, y, size, size).data;

    // Check if all pixels are identical
    const [r0, g0, b0] = [data[0], data[1], data[2]];
    for (let i = 4; i < data.length; i += 4) {
      if (data[i] !== r0 || data[i + 1] !== g0 || data[i + 2] !== b0) {
        return true; // Non-blank
      }
    }
    return false;
  }, selector);
}

/**
 * Compute SDR (dB) between the original and reconstructed signals
 * using the `__e2e` data exposed by the app.
 *
 * Returns { sdrLeft, sdrRight } or null if __e2e is not available.
 */
export async function computeStemSumSDR(
  page: Page,
): Promise<{ sdrLeft: number; sdrRight: number } | null> {
  return page.evaluate(() => {
    const e2e = (window as any).__e2e;
    if (!e2e) return null;

    const { stems, originalLeft, originalRight } = e2e as {
      stems: Array<{ left: Float32Array; right: Float32Array }>;
      originalLeft: Float32Array;
      originalRight: Float32Array;
    };

    const n = originalLeft.length;

    // Sum all stem channels
    const sumL = new Float32Array(n);
    const sumR = new Float32Array(n);
    for (const stem of stems) {
      const len = Math.min(stem.left.length, n);
      for (let i = 0; i < len; i++) {
        sumL[i] += stem.left[i];
        sumR[i] += stem.right[i];
      }
    }

    // Compute SDR
    function sdr(ref_: Float32Array, est: Float32Array): number {
      let signal = 0;
      let noise = 0;
      for (let i = 0; i < ref_.length; i++) {
        signal += ref_[i] * ref_[i];
        const diff = ref_[i] - est[i];
        noise += diff * diff;
      }
      if (noise < 1e-20) return 100;
      return 10 * Math.log10(signal / noise);
    }

    return {
      sdrLeft: sdr(originalLeft, sumL),
      sdrRight: sdr(originalRight, sumR),
    };
  });
}
