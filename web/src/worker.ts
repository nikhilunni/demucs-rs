import initWasm, { compute_spectrogram, get_model_registry, separate } from "./wasm/demucs_wasm.js";

let wasmReady: Promise<any>;

self.onmessage = async (e: MessageEvent) => {
  const { type, id, ...data } = e.data;

  try {
    switch (type) {
      case "init": {
        wasmReady = initWasm(data.wasmUrl);
        await wasmReady;
        const registry = get_model_registry();
        self.postMessage({ id, type: "init", registry });
        break;
      }

      case "spectrogram": {
        await wasmReady;
        const result = compute_spectrogram(data.samples);
        // Read getters before take_mags() which consumes the result
        const numFrames = result.num_frames;
        const numBins = result.num_bins;
        const mags = result.take_mags();
        self.postMessage(
          { id, type: "spectrogram", mags, numFrames, numBins },
          { transfer: [mags.buffer] },
        );
        break;
      }

      case "separate": {
        await wasmReady;

        // Progress callback: forward events to main thread (fire-and-forget)
        const onProgress = (event: any) => {
          self.postMessage({ type: "progress", event });
        };

        const result = await separate(
          data.modelBytes,
          data.modelId,
          data.stems,
          data.left,
          data.right,
          data.sampleRate,
          onProgress,
        );
        // Read getters before take_audio() which consumes the result
        const stemNames: string[] = result.stem_names();
        const nSamples = result.n_samples;
        const numStems = result.num_stems;
        const audio = result.take_audio();
        self.postMessage(
          { id, type: "separate", audio, stemNames, nSamples, numStems },
          { transfer: [audio.buffer] },
        );
        break;
      }
    }
  } catch (err) {
    self.postMessage({
      id,
      type: "error",
      error: err instanceof Error ? err.message : String(err),
    });
  }
};
