/**
 * Model registry — derived from Rust metadata via WASM.
 *
 * Call `initRegistry()` after WASM is initialized to populate the model list.
 * Before that, `getModels()` will throw.
 */

export type StemId = "drums" | "bass" | "other" | "vocals" | "guitar" | "piano";

export interface ModelVariant {
  id: string;
  label: string;
  description: string;
  sizeMb: number;
  stems: StemId[];
  filename: string;
  downloadUrl: string;
}

export interface SelectedModel {
  variant: ModelVariant;
  stems: StemId[];
}

/** Populated by `initRegistry()` after WASM loads. */
let _models: ModelVariant[] = [];

/** Raw WASM model info shape (matches JsModelInfo in Rust). */
interface WasmModelInfo {
  id: string;
  label: string;
  description: string;
  size_mb: number;
  stems: string[];
  filename: string;
  download_url: string;
}

/**
 * Initialize the model registry from WASM metadata.
 * Must be called once after WASM is loaded, before rendering the model sidebar.
 */
export function initRegistry(wasmModels: WasmModelInfo[]): void {
  _models = wasmModels.map((m) => ({
    id: m.id,
    label: m.label,
    description: m.description,
    sizeMb: m.size_mb,
    stems: m.stems as StemId[],
    filename: m.filename,
    downloadUrl: m.download_url,
  }));
}

/** Get the model list. Throws if registry not initialized. */
export function getModels(): readonly ModelVariant[] {
  if (_models.length === 0) {
    throw new Error("Model registry not initialized — call initRegistry() after WASM loads");
  }
  return _models;
}

export function modelUrl(m: ModelVariant): string {
  return m.downloadUrl;
}
