/**
 * Static model definitions for HTDemucs variants.
 */

export type StemId = "drums" | "bass" | "other" | "vocals" | "guitar" | "piano";

export interface ModelVariant {
  id: string;
  label: string;
  description: string;
  sizeMb: number;
  stems: StemId[];
  filename: string;
}

export const MODELS: readonly ModelVariant[] = [
  {
    id: "htdemucs",
    label: "Standard",
    description: "4 stems — balanced speed and quality",
    sizeMb: 84,
    stems: ["drums", "bass", "other", "vocals"],
    filename: "htdemucs.safetensors",
  },
  {
    id: "htdemucs_6s",
    label: "6-Stem",
    description: "6 stems — adds guitar and piano",
    sizeMb: 55,
    stems: ["drums", "bass", "other", "vocals", "guitar", "piano"],
    filename: "htdemucs_6s.safetensors",
  },
  {
    id: "htdemucs_ft",
    label: "Fine-Tuned",
    description: "4 stems — best quality, larger download",
    sizeMb: 336,
    stems: ["drums", "bass", "other", "vocals"],
    filename: "htdemucs_ft.safetensors",
  },
] as const;

export const HF_BASE =
  "https://huggingface.co/set-soft/audio_separation/resolve/main/Demucs/";

export function modelUrl(m: ModelVariant): string {
  return `${HF_BASE}${m.filename}`;
}

export interface SelectedModel {
  variant: ModelVariant;
  stems: StemId[];
}
