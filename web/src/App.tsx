import { useCallback, useEffect, useRef, useState } from "react";
import { DropZone } from "./components/DropZone";
import { LoadingState } from "./components/LoadingState";
import { WarmupScreen } from "./components/WarmupScreen";
import { ModelSidebar } from "./components/ModelSidebar";
import { SpectrogramView } from "./components/SpectrogramView";
import { PlayerControls } from "./components/PlayerControls";
import { StemResults, type StemData } from "./components/StemResults";
import {
  ModelProgress,
  reduceProgress,
  INITIAL_PROGRESS,
  type ModelProgressState,
} from "./components/ModelProgress";
import { useAudioPlayer } from "./hooks/useAudioPlayer";
import { useMultiTrackPlayer, type StemTrack } from "./hooks/useMultiTrackPlayer";
import { useDemucsWorker, type ProgressEvent } from "./hooks/useDemucsWorker";
import { renderSpectrogramImage } from "./dsp/colormap";
import { encodeWavUrl } from "./dsp/wav";
import { isCached, loadModel } from "./models/modelCache";
import type { SelectedModel } from "./models/registry";
import { initRegistry } from "./models/registry";
import wasmUrl from "./wasm/demucs_wasm_bg.wasm?url";
import "./App.css";

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
  | { kind: "warmup" }
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready" }
  | { kind: "separating" }
  | { kind: "separated"; stems: StemData[]; tracks: StemTrack[] };

export default function App() {
  const [phase, setPhase] = useState<Phase>({ kind: "warmup" });
  const [trackInfo, setTrackInfo] = useState<TrackInfo | null>(null);
  const [progress, setProgress] = useState<ModelProgressState>(INITIAL_PROGRESS);
  const [clipRange, setClipRange] = useState<[number, number] | null>(null);

  const audioUrl = trackInfo?.audioUrl ?? null;
  const player = useAudioPlayer(audioUrl);

  const stemTracks = phase.kind === "separated" ? phase.tracks : null;
  const multiPlayer = useMultiTrackPlayer(stemTracks);

  // Keep a ref to current stem blob URLs for cleanup
  const stemUrlsRef = useRef<string[]>([]);

  // Worker for off-main-thread WASM execution
  const worker = useDemucsWorker();
  const initRef = useRef<Promise<void> | undefined>(undefined);

  // Initialize WASM in the worker on mount, then warmup GPU if a model is cached
  useEffect(() => {
    const DEFAULT_MODEL = "htdemucs";

    initRef.current = worker.init(wasmUrl).then(async ({ registry }) => {
      initRegistry(registry);

      // If a model is already cached, warmup GPU shaders with a dummy forward pass
      const cached = await isCached(DEFAULT_MODEL);
      if (cached) {
        const bytes = await loadModel(DEFAULT_MODEL);
        if (bytes) {
          try {
            await worker.warmup({
              modelBytes: new Uint8Array(bytes),
              modelId: DEFAULT_MODEL,
            });
          } catch (err) {
            console.warn("GPU warmup failed (non-fatal):", err);
          }
        }
      }
      setPhase({ kind: "idle" });
    });
  }, [worker]);

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
    setClipRange(null);

    // Yield to let React paint the loading state before heavy computation
    await new Promise((r) => setTimeout(r, 60));

    try {
      await initRef.current;

      const url = URL.createObjectURL(file);

      const arrayBuf = await file.arrayBuffer();
      const audioCtx = new AudioContext({ sampleRate: 44100 });
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

      // Compute spectrogram in worker (transfers mono buffer)
      const { mags, numFrames, numBins } = await worker.spectrogram(mono);
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
  }, [phase, trackInfo, worker]);

  // Stable ref for progress callback (avoids recreating closure)
  const progressRef = useRef<ModelProgressState>(INITIAL_PROGRESS);

  const handleRun = useCallback(async (selection: SelectedModel) => {
    if (!trackInfo) return;

    setProgress(INITIAL_PROGRESS);
    progressRef.current = INITIAL_PROGRESS;
    setPhase({ kind: "separating" });
    await new Promise((r) => setTimeout(r, 60));

    try {
      // Load model bytes from IndexedDB
      const bytes = await loadModel(selection.variant.id);
      if (!bytes) throw new Error("Model not found in cache");

      const modelBytes = new Uint8Array(bytes);
      const stemNames = selection.stems.map((s) => s as string);

      // Progress callback — accumulate state and push to React
      const onProgress = (event: ProgressEvent) => {
        const next = reduceProgress(progressRef.current, event);
        progressRef.current = next;
        setProgress(next);
      };

      // Slice audio to clip range if set
      let left = trackInfo.left;
      let right = trackInfo.right;
      if (clipRange) {
        const s = Math.floor(clipRange[0] * trackInfo.sampleRate);
        const e = Math.ceil(clipRange[1] * trackInfo.sampleRate);
        left = left.slice(s, e);
        right = right.slice(s, e);
      }

      // Run separation in worker (transfers modelBytes buffer)
      const { audio, stemNames: names, nSamples, numStems } = await worker.separate(
        {
          modelBytes,
          modelId: selection.variant.id,
          stems: stemNames,
          left,
          right,
          sampleRate: trackInfo.sampleRate,
        },
        onProgress,
      );

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

        // Compute spectrogram for this stem in worker
        const spec = await worker.spectrogram(mono);
        const image = renderSpectrogramImage(
          { mags: spec.mags, numFrames: spec.numFrames, numBins: spec.numBins },
          trackInfo.sampleRate,
        );

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
  }, [trackInfo, worker, clipRange]);

  const handleReset = useCallback(() => {
    if (trackInfo) {
      URL.revokeObjectURL(trackInfo.audioUrl);
    }
    for (const url of stemUrlsRef.current) URL.revokeObjectURL(url);
    stemUrlsRef.current = [];
    setTrackInfo(null);
    setClipRange(null);
    setPhase({ kind: "idle" });
  }, [trackInfo]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Space → play / pause
      if (e.code === "Space") {
        if (!trackInfo) return;
        e.preventDefault();
        if (phase.kind === "separated") {
          multiPlayer.toggleAll();
        } else {
          player.toggle();
        }
        return;
      }

      // Escape → clear clip selection
      if (e.code === "Escape" && clipRange) {
        setClipRange(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [phase.kind, trackInfo, player.toggle, multiPlayer.toggleAll, clipRange]);

  const hasTrack = trackInfo !== null;
  const tagline =
    phase.kind === "warmup"
      ? "Initializing..."
      : phase.kind === "idle"
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
        {phase.kind === "warmup" && <WarmupScreen />}
        {phase.kind === "idle" && <DropZone onFile={handleFile} />}
        {phase.kind === "loading" && <LoadingState />}
        {hasTrack && (
          <>
            <div className="player-area">
              <SpectrogramView
                image={trackInfo.image}
                currentTime={
                  phase.kind === "separated" && clipRange
                    ? clipRange[0] + multiPlayer.currentTime
                    : phase.kind === "separated"
                      ? multiPlayer.currentTime
                      : player.currentTime
                }
                duration={player.duration}
                sampleRate={trackInfo.sampleRate}
                onSeek={
                  phase.kind === "separated" && clipRange
                    ? (time: number) => {
                        const clamped = Math.max(clipRange[0], Math.min(clipRange[1], time));
                        multiPlayer.seek(clamped - clipRange[0]);
                      }
                    : phase.kind === "separated"
                      ? multiPlayer.seek
                      : player.seek
                }
                clipRange={clipRange}
                onClipChange={phase.kind === "separated" ? undefined : setClipRange}
              />
              <PlayerControls
                fileName={trackInfo.fileName}
                isPlaying={phase.kind === "separated" ? multiPlayer.isPlaying : player.isPlaying}
                currentTime={phase.kind === "separated" ? multiPlayer.currentTime : player.currentTime}
                duration={phase.kind === "separated" ? multiPlayer.duration : player.duration}
                onToggle={phase.kind === "separated" ? multiPlayer.toggleAll : player.toggle}
              />

              {phase.kind === "separating" && (
                <ModelProgress progress={progress} />
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
              clipRange={clipRange}
            />
          </>
        )}
      </main>
    </div>
  );
}
