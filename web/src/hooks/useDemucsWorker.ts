import { useCallback, useEffect, useMemo, useRef } from "react";

export interface SpectrogramResult {
  mags: Float32Array;
  numFrames: number;
  numBins: number;
}

export interface SeparateResult {
  audio: Float32Array;
  stemNames: string[];
  nSamples: number;
  numStems: number;
}

export interface SeparateOptions {
  modelBytes: Uint8Array;
  modelId: string;
  stems: string[];
  left: Float32Array;
  right: Float32Array;
  sampleRate: number;
}

/** Progress events emitted during model forward pass. */
export type ProgressEvent =
  | { type: "chunk_started"; index: number; total: number }
  | { type: "chunk_done"; index: number; total: number }
  | { type: "encoder_done"; domain: "freq" | "time"; layer: number; numLayers: number }
  | { type: "transformer_done" }
  | { type: "decoder_done"; domain: "freq" | "time"; layer: number; numLayers: number }
  | { type: "denormalized" }
  | { type: "stem_done"; index: number; total: number };

export interface WarmupOptions {
  modelBytes: Uint8Array;
  modelId: string;
}

export interface DemucsWorker {
  init(wasmUrl: string): Promise<{ registry: any }>;
  warmup(opts: WarmupOptions): Promise<void>;
  spectrogram(samples: Float32Array): Promise<SpectrogramResult>;
  separate(opts: SeparateOptions, onProgress?: (event: ProgressEvent) => void): Promise<SeparateResult>;
}

/**
 * React hook that manages a Web Worker running WASM inference.
 * Provides a promise-based API for spectrogram computation and source separation.
 */
export function useDemucsWorker(): DemucsWorker {
  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef(
    new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>(),
  );
  const idRef = useRef(0);
  const progressRef = useRef<((event: ProgressEvent) => void) | null>(null);

  useEffect(() => {
    const worker = new Worker(
      new URL("../worker.ts", import.meta.url),
      { type: "module" },
    );
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent) => {
      const { id, type, ...data } = e.data;

      // Progress events are fire-and-forget (no id)
      if (type === "progress") {
        progressRef.current?.(data.event);
        return;
      }

      const pending = pendingRef.current.get(id);
      if (!pending) return;
      pendingRef.current.delete(id);

      if (type === "error") {
        pending.reject(new Error(data.error));
      } else {
        pending.resolve(data);
      }
    };

    return () => {
      worker.terminate();
      workerRef.current = null;
      // Reject any pending requests
      for (const [, p] of pendingRef.current) {
        p.reject(new Error("Worker terminated"));
      }
      pendingRef.current.clear();
    };
  }, []);

  const send = useCallback(
    (type: string, data: Record<string, any>, transfer: Transferable[] = []) => {
      return new Promise<any>((resolve, reject) => {
        const worker = workerRef.current;
        if (!worker) {
          reject(new Error("Worker not ready"));
          return;
        }
        const id = idRef.current++;
        pendingRef.current.set(id, { resolve, reject });
        worker.postMessage({ type, id, ...data }, transfer);
      });
    },
    [],
  );

  const init = useCallback(
    (wasmUrl: string) => send("init", { wasmUrl }),
    [send],
  );

  const spectrogram = useCallback(
    (samples: Float32Array) =>
      send("spectrogram", { samples }, [samples.buffer]),
    [send],
  );

  const warmup = useCallback(
    (opts: WarmupOptions) =>
      send("warmup", opts, [opts.modelBytes.buffer]),
    [send],
  );

  const separate = useCallback(
    (opts: SeparateOptions, onProgress?: (event: ProgressEvent) => void) => {
      progressRef.current = onProgress ?? null;
      return send("separate", opts, [opts.modelBytes.buffer]);
    },
    [send],
  );

  return useMemo(
    () => ({ init, warmup, spectrogram, separate }),
    [init, warmup, spectrogram, separate],
  );
}
