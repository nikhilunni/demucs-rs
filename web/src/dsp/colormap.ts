/**
 * Spectrogram colormap and image renderer.
 *
 * The colormap is a 256-entry RGBA lookup table built from the magma-inspired
 * stops defined in design/tokens.ts. The renderer maps STFT dB magnitudes to
 * pixel colors with log-frequency Y-axis scaling.
 */
import { colormapStops, stftParams, spectrogramDisplay } from "../design/tokens";

export interface StftResult {
  mags: Float32Array;
  numFrames: number;
  numBins: number;
}

const { nFft: N_FFT } = stftParams;
const { renderHeight: HEIGHT, minFreq: MIN_FREQ, dynamicRange: DYN_RANGE } =
  spectrogramDisplay;

/* ── Build the 256-entry RGBA lookup table ────────────────────── */

function buildColormap(): Uint8Array {
  const lut = new Uint8Array(256 * 4);

  for (let i = 0; i < 256; i++) {
    const t = i / 255;

    // Find the two surrounding stops
    let lo = 0;
    for (let s = 0; s < colormapStops.length - 1; s++) {
      if (t >= colormapStops[s][0]) lo = s;
    }
    const hi = Math.min(lo + 1, colormapStops.length - 1);

    const range = colormapStops[hi][0] - colormapStops[lo][0];
    const frac = range > 0 ? (t - colormapStops[lo][0]) / range : 0;

    const idx = i * 4;
    lut[idx] = Math.round(
      colormapStops[lo][1] + (colormapStops[hi][1] - colormapStops[lo][1]) * frac,
    );
    lut[idx + 1] = Math.round(
      colormapStops[lo][2] + (colormapStops[hi][2] - colormapStops[lo][2]) * frac,
    );
    lut[idx + 2] = Math.round(
      colormapStops[lo][3] + (colormapStops[hi][3] - colormapStops[lo][3]) * frac,
    );
    lut[idx + 3] = 255;
  }

  return lut;
}

export const COLORMAP = buildColormap();

/* ── Render spectrogram to an offscreen canvas ────────────────── */

/**
 * Produces an HTMLCanvasElement of size `[numFrames × renderHeight]`
 * with log-frequency Y-axis and the magma colormap applied.
 */
export function renderSpectrogramImage(
  stft: StftResult,
  sampleRate: number,
): HTMLCanvasElement {
  const { mags, numFrames, numBins } = stft;

  // Global dB range
  let dbMin = Infinity;
  let dbMax = -Infinity;
  for (let i = 0; i < mags.length; i++) {
    if (mags[i] < dbMin) dbMin = mags[i];
    if (mags[i] > dbMax) dbMax = mags[i];
  }
  dbMin = Math.max(dbMin, dbMax - DYN_RANGE);
  const dbRange = dbMax - dbMin || 1;

  const canvas = document.createElement("canvas");
  canvas.width = numFrames;
  canvas.height = HEIGHT;

  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(numFrames, HEIGHT);
  const px = img.data;

  const nyquist = sampleRate / 2;
  const logMin = Math.log10(MIN_FREQ);
  const logMax = Math.log10(nyquist);
  const logRange = logMax - logMin;

  for (let y = 0; y < HEIGHT; y++) {
    // Log-frequency mapping: top = high freq, bottom = low freq
    const logFreq = logMax - (y / (HEIGHT - 1)) * logRange;
    const freq = 10 ** logFreq;
    const binF = (freq * N_FFT) / sampleRate;
    const binLo = Math.floor(binF);
    const binHi = Math.min(binLo + 1, numBins - 1);
    const frac = binF - binLo;

    for (let x = 0; x < numFrames; x++) {
      const base = x * numBins;
      // Linear interpolation between adjacent frequency bins
      const db = mags[base + binLo] * (1 - frac) + mags[base + binHi] * frac;
      const norm = Math.max(0, Math.min(1, (db - dbMin) / dbRange));
      const ci = Math.round(norm * 255) * 4;
      const pi = (y * numFrames + x) * 4;
      px[pi] = COLORMAP[ci];
      px[pi + 1] = COLORMAP[ci + 1];
      px[pi + 2] = COLORMAP[ci + 2];
      px[pi + 3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);
  return canvas;
}
