/**
 * Design tokens for Demucs UI.
 *
 * These values define the shared visual language between the web app (React)
 * and the future VST plugin (egui). When building the egui UI, translate
 * these constants into egui::Color32 / egui::Stroke / etc.
 */

/* ── Color palette ────────────────────────────────────────────── */

export const palette = {
  bg: "#0b0a10",
  surface: "#14131f",
  surface2: "#1e1d2e",
  border: "#2a2940",

  text: "#e8e5f0",
  textDim: "#6e6b82",
  textMicro: "#4a4860",

  /** Warm accent — pulled from the magma colormap mid-highs */
  accent: "#f07a5c",
  accentGlow: "rgba(240, 122, 92, 0.35)",

  /** Cool accent — indigo/purple secondary */
  accentCool: "#7c6ff0",
  accentCoolGlow: "rgba(124, 111, 240, 0.35)",
} as const;

/* ── Typography ───────────────────────────────────────────────── */

export const fonts = {
  display: "'Syne', sans-serif",
  body: "'Outfit', sans-serif",
  mono: "'JetBrains Mono', monospace",
} as const;

/* ── Spacing ──────────────────────────────────────────────────── */

export const space = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
} as const;

/* ── Spectrogram colormap stops (magma-inspired) ──────────────── */
/* Each entry: [position 0..1, r, g, b]                           */

export const colormapStops: readonly [number, number, number, number][] = [
  [0.0, 0, 0, 4],
  [0.1, 20, 14, 54],
  [0.2, 59, 15, 112],
  [0.3, 100, 26, 128],
  [0.4, 140, 41, 129],
  [0.5, 183, 55, 121],
  [0.6, 222, 73, 104],
  [0.7, 247, 115, 92],
  [0.8, 254, 176, 120],
  [0.9, 253, 226, 163],
  [1.0, 252, 253, 191],
];

/* ── STFT parameters (must match demucs-core/src/stft.rs) ─────── */

export const stftParams = {
  nFft: 4096,
  hopLength: 1024,
  get numBins() {
    return this.nFft / 2 + 1;
  },
} as const;

/* ── Spectrogram display ──────────────────────────────────────── */

export const spectrogramDisplay = {
  /** Pixel height of the offscreen spectrogram render */
  renderHeight: 512,
  /** Minimum displayed frequency (Hz) */
  minFreq: 30,
  /** Dynamic range in dB */
  dynamicRange: 80,
  /** Frequency tick marks for the Y axis (Hz) */
  freqTicks: [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
} as const;
