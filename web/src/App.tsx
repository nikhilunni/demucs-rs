import { useCallback, useEffect, useRef, useState } from "react";
import { DropZone } from "./components/DropZone";
import { LoadingState } from "./components/LoadingState";
import { ModelSidebar } from "./components/ModelSidebar";
import { SpectrogramView } from "./components/SpectrogramView";
import { PlayerControls } from "./components/PlayerControls";
import { StemResults, type StemData } from "./components/StemResults";
import { useAudioPlayer } from "./hooks/useAudioPlayer";
import { useMultiTrackPlayer, type StemTrack } from "./hooks/useMultiTrackPlayer";
import { renderSpectrogramImage } from "./dsp/colormap";
import { encodeWavUrl } from "./dsp/wav";
import { loadModel } from "./models/modelCache";
import type { SelectedModel } from "./models/registry";
import initWasm, { compute_spectrogram, get_model_registry, separate } from "./wasm/demucs_wasm.js";
import wasmUrl from "./wasm/demucs_wasm_bg.wasm?url";
import { initRegistry } from "./models/registry";
import "./App.css";

// Start loading WASM immediately on module import;
// populate the model registry as soon as it's ready.
const wasmReady = initWasm(wasmUrl).then(() => {
  initRegistry(get_model_registry());
});

/** Persistent info about the loaded track (survives phase transitions). */
interface TrackInfo {
  image: HTMLCanvasElement;
  audioUrl: string;
  fileName: string;
  sampleRate: number;
  left: Float32Array;
  right: Float32Array;
}

type Phase =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready" }
  | { kind: "separating" }
  | { kind: "separated"; stems: StemData[]; tracks: StemTrack[] };

export default function App() {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [trackInfo, setTrackInfo] = useState<TrackInfo | null>(null);

  const audioUrl = trackInfo?.audioUrl ?? null;
  const player = useAudioPlayer(audioUrl);

  const stemTracks = phase.kind === "separated" ? phase.tracks : null;
  const multiPlayer = useMultiTrackPlayer(stemTracks);

  // Keep a ref to current stem blob URLs for cleanup
  const stemUrlsRef = useRef<string[]>([]);

  const handleFile = useCallback(async (file: File) => {
    if (phase.kind === "loading") return;

    // Revoke previous URLs
    if (trackInfo) {
      URL.revokeObjectURL(trackInfo.audioUrl);
    }
    for (const url of stemUrlsRef.current) URL.revokeObjectURL(url);
    stemUrlsRef.current = [];

    setPhase({ kind: "loading" });
    setTrackInfo(null);

    // Yield to let React paint the loading state before heavy computation
    await new Promise((r) => setTimeout(r, 60));

    try {
      await wasmReady;

      const url = URL.createObjectURL(file);

      const arrayBuf = await file.arrayBuffer();
      const audioCtx = new AudioContext();
      const audioBuf = await audioCtx.decodeAudioData(arrayBuf);
      const left = audioBuf.getChannelData(0);
      // Use right channel if stereo, otherwise duplicate mono
      const right = audioBuf.numberOfChannels >= 2
        ? audioBuf.getChannelData(1)
        : audioBuf.getChannelData(0);
      const sr = audioBuf.sampleRate;
      audioCtx.close();

      // Mono-mix for spectrogram
      const mono = new Float32Array(left.length);
      for (let i = 0; i < left.length; i++) {
        mono[i] = (left[i] + right[i]) * 0.5;
      }

      const result = compute_spectrogram(mono);
      const numFrames = result.num_frames;
      const numBins = result.num_bins;
      const mags = result.take_mags();
      const image = renderSpectrogramImage({ mags, numFrames, numBins }, sr);

      setTrackInfo({
        image,
        audioUrl: url,
        fileName: file.name,
        sampleRate: sr,
        left: new Float32Array(left),
        right: new Float32Array(right),
      });
      setPhase({ kind: "ready" });
    } catch (err) {
      console.error("Failed to load audio:", err);
      alert("Could not decode that file — is it a valid audio file?");
      setPhase({ kind: "idle" });
    }
  }, [phase, trackInfo]);

  const handleRun = useCallback(async (selection: SelectedModel) => {
    if (!trackInfo) return;

    setPhase({ kind: "separating" });
    await new Promise((r) => setTimeout(r, 60));

    try {
      // Load model bytes from IndexedDB
      const bytes = await loadModel(selection.variant.id);
      if (!bytes) throw new Error("Model not found in cache");

      const modelBytes = new Uint8Array(bytes);
      const stemNames = selection.stems.map((s) => s as string);

      // Run separation via WASM
      const result = separate(
        modelBytes,
        selection.variant.id,
        stemNames,
        trackInfo.left,
        trackInfo.right,
      );

      const numStems = result.num_stems;
      const nSamples = result.n_samples;
      const names: string[] = result.stem_names();
      const audio = result.take_audio();

      // Build per-stem data
      const stems: StemData[] = [];
      const tracks: StemTrack[] = [];
      const urls: string[] = [];

      for (let i = 0; i < numStems; i++) {
        const offset = i * 2 * nSamples;
        const stemLeft = new Float32Array(audio.buffer, offset * 4, nSamples);
        const stemRight = new Float32Array(audio.buffer, (offset + nSamples) * 4, nSamples);

        // Mono-mix for spectrogram
        const mono = new Float32Array(nSamples);
        for (let j = 0; j < nSamples; j++) {
          mono[j] = (stemLeft[j] + stemRight[j]) * 0.5;
        }

        // Compute spectrogram for this stem
        const spec = compute_spectrogram(mono);
        const numFrames = spec.num_frames;
        const numBins = spec.num_bins;
        const mags = spec.take_mags();
        const image = renderSpectrogramImage({ mags, numFrames, numBins }, trackInfo.sampleRate);

        // Encode WAV blob URL
        const wavUrl = encodeWavUrl(stemLeft, stemRight, trackInfo.sampleRate);
        urls.push(wavUrl);

        stems.push({ name: names[i], image, url: wavUrl });
        tracks.push({ name: names[i], url: wavUrl });
      }

      stemUrlsRef.current = urls;
      setPhase({ kind: "separated", stems, tracks });
    } catch (err) {
      console.error("Separation failed:", err);
      alert(`Separation failed: ${err instanceof Error ? err.message : err}`);
      setPhase({ kind: "ready" });
    }
  }, [trackInfo]);

  const handleReset = useCallback(() => {
    if (trackInfo) {
      URL.revokeObjectURL(trackInfo.audioUrl);
    }
    for (const url of stemUrlsRef.current) URL.revokeObjectURL(url);
    stemUrlsRef.current = [];
    setTrackInfo(null);
    setPhase({ kind: "idle" });
  }, [trackInfo]);

  // Space bar → play / pause (multi-track when separated, original otherwise)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.code !== "Space") return;
      if (!trackInfo) return;
      e.preventDefault();

      if (phase.kind === "separated") {
        multiPlayer.toggleAll();
      } else {
        player.toggle();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [phase.kind, trackInfo, player.toggle, multiPlayer.toggleAll]);

  const hasTrack = trackInfo !== null;
  const tagline =
    phase.kind === "idle"
      ? "Source Separation"
      : phase.kind === "loading"
        ? "Analyzing..."
        : phase.kind === "separating"
          ? "Separating..."
          : trackInfo?.fileName ?? "";

  return (
    <div className="app">
      <header className="header">
        <h1 className="logo">Demucs</h1>
        <span className="tagline">{tagline}</span>
        {hasTrack && (
          <button
            className="reset-btn"
            onClick={handleReset}
            title="Load a different file"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        )}
      </header>

      <main className={`main ${hasTrack ? "main--player" : ""}`}>
        {phase.kind === "idle" && <DropZone onFile={handleFile} />}
        {phase.kind === "loading" && <LoadingState />}
        {hasTrack && (
          <>
            <div className="player-area">
              <SpectrogramView
                image={trackInfo.image}
                currentTime={phase.kind === "separated" ? multiPlayer.currentTime : player.currentTime}
                duration={phase.kind === "separated" ? multiPlayer.duration : player.duration}
                sampleRate={trackInfo.sampleRate}
                onSeek={phase.kind === "separated" ? multiPlayer.seek : player.seek}
              />
              <PlayerControls
                fileName={trackInfo.fileName}
                isPlaying={phase.kind === "separated" ? multiPlayer.isPlaying : player.isPlaying}
                currentTime={phase.kind === "separated" ? multiPlayer.currentTime : player.currentTime}
                duration={phase.kind === "separated" ? multiPlayer.duration : player.duration}
                onToggle={phase.kind === "separated" ? multiPlayer.toggleAll : player.toggle}
              />

              {phase.kind === "separating" && (
                <div className="separating-indicator">
                  <div className="loading-bars">
                    <div className="loading-bar" style={{ animationDelay: "0s" }} />
                    <div className="loading-bar" style={{ animationDelay: "0.15s" }} />
                    <div className="loading-bar" style={{ animationDelay: "0.3s" }} />
                    <div className="loading-bar" style={{ animationDelay: "0.45s" }} />
                    <div className="loading-bar" style={{ animationDelay: "0.6s" }} />
                  </div>
                  <span className="loading-text">Separating stems...</span>
                </div>
              )}

              {phase.kind === "separated" && (
                <StemResults
                  stems={phase.stems}
                  player={multiPlayer}
                  sampleRate={trackInfo.sampleRate}
                />
              )}
            </div>
            <ModelSidebar
              onRun={handleRun}
              disabled={phase.kind === "separating"}
            />
          </>
        )}
      </main>
    </div>
  );
}
