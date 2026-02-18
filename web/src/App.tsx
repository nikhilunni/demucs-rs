import { useCallback, useEffect, useState } from "react";
import { DropZone } from "./components/DropZone";
import { LoadingState } from "./components/LoadingState";
import { SpectrogramView } from "./components/SpectrogramView";
import { PlayerControls } from "./components/PlayerControls";
import { useAudioPlayer } from "./hooks/useAudioPlayer";
import { renderSpectrogramImage } from "./dsp/colormap";
import initWasm, { compute_spectrogram } from "./wasm/demucs_wasm.js";
import wasmUrl from "./wasm/demucs_wasm_bg.wasm?url";
import "./App.css";

// Start loading WASM immediately on module import
const wasmReady = initWasm(wasmUrl);

type Phase =
  | { kind: "idle" }
  | { kind: "loading" }
  | {
      kind: "ready";
      image: HTMLCanvasElement;
      audioUrl: string;
      fileName: string;
      sampleRate: number;
    };

export default function App() {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });

  const audioUrl = phase.kind === "ready" ? phase.audioUrl : null;
  const player = useAudioPlayer(audioUrl);

  const handleFile = useCallback(async (file: File) => {
    setPhase({ kind: "loading" });

    // Yield to let React paint the loading state before heavy computation
    await new Promise((r) => setTimeout(r, 60));

    try {
      // Ensure WASM is ready
      await wasmReady;

      // Create playback URL from the File (which is already a Blob)
      const url = URL.createObjectURL(file);

      // Decode to raw samples for the STFT
      const arrayBuf = await file.arrayBuffer();
      const audioCtx = new AudioContext();
      const audioBuf = await audioCtx.decodeAudioData(arrayBuf);
      const samples = audioBuf.getChannelData(0);
      const sr = audioBuf.sampleRate;
      audioCtx.close();

      // Compute spectrogram via Rust WASM
      const result = compute_spectrogram(samples);
      const numFrames = result.num_frames;
      const numBins = result.num_bins;
      const mags = result.take_mags(); // consumes the WASM struct
      const image = renderSpectrogramImage({ mags, numFrames, numBins }, sr);

      setPhase({ kind: "ready", image, audioUrl: url, fileName: file.name, sampleRate: sr });
    } catch (err) {
      console.error("Failed to load audio:", err);
      alert("Could not decode that file — is it a valid audio file?");
      setPhase({ kind: "idle" });
    }
  }, []);

  // Space bar → play / pause
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.code === "Space" && phase.kind === "ready") {
        e.preventDefault();
        player.toggle();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [phase.kind, player.toggle]);

  return (
    <div className="app">
      <header className="header">
        <h1 className="logo">Demucs</h1>
        <span className="tagline">Spectrogram Viewer</span>
      </header>

      <main className={`main ${phase.kind === "ready" ? "main--player" : ""}`}>
        {phase.kind === "idle" && <DropZone onFile={handleFile} />}
        {phase.kind === "loading" && <LoadingState />}
        {phase.kind === "ready" && (
          <div className="player">
            <SpectrogramView
              image={phase.image}
              currentTime={player.currentTime}
              duration={player.duration}
              sampleRate={phase.sampleRate}
              onSeek={player.seek}
            />
            <PlayerControls
              fileName={phase.fileName}
              isPlaying={player.isPlaying}
              currentTime={player.currentTime}
              duration={player.duration}
              onToggle={player.toggle}
            />
          </div>
        )}
      </main>
    </div>
  );
}
