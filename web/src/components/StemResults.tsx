import { SpectrogramView } from "./SpectrogramView";
import type { MultiTrackPlayer } from "../hooks/useMultiTrackPlayer";

export interface StemData {
  name: string;
  image: HTMLCanvasElement;
  url: string; // blob URL for playback
}

const STEM_COLORS: Record<string, string> = {
  drums: "#f0a05c",
  bass: "#b07af0",
  other: "#6ee7a0",
  vocals: "#f0d75c",
  guitar: "#5cb8f0",
  piano: "#f07a7a",
};

interface Props {
  stems: StemData[];
  player: MultiTrackPlayer;
  sampleRate: number;
}

export function StemResults({ stems, player, sampleRate }: Props) {
  return (
    <div className="stem-results">
      <button className="stem-results__play-all" onClick={player.toggleAll}>
        {player.isPlaying ? "Pause All" : "Play All"}
      </button>

      {stems.map((stem) => (
        <StemRow
          key={stem.name}
          stem={stem}
          player={player}
          sampleRate={sampleRate}
        />
      ))}
    </div>
  );
}

function StemRow({
  stem,
  player,
  sampleRate,
}: {
  stem: StemData;
  player: MultiTrackPlayer;
  sampleRate: number;
}) {
  const isMuted = player.muted[stem.name] ?? false;
  const isSolo = player.solo === stem.name;
  const color = STEM_COLORS[stem.name] ?? "var(--text)";

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = stem.url;
    a.download = `${stem.name}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="stem-row">
      <div className="stem-row__controls">
        <span className="stem-row__name" style={{ color }}>
          {stem.name}
        </span>
        <div className="stem-row__buttons">
          <button
            className={`stem-row__solo ${isSolo ? "stem-row__solo--active" : ""}`}
            onClick={() => player.playSolo(stem.name)}
            title="Solo"
          >
            S
          </button>
          <button
            className={`stem-row__mute ${isMuted ? "stem-row__mute--active" : ""}`}
            onClick={() => player.toggleMute(stem.name)}
            title="Mute"
          >
            M
          </button>
          <button
            className="stem-row__download"
            onClick={handleDownload}
            title="Download WAV"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
          </button>
        </div>
      </div>
      <div className="stem-row__spectrogram">
        <SpectrogramView
          image={stem.image}
          currentTime={player.currentTime}
          duration={player.duration}
          sampleRate={sampleRate}
          onSeek={player.seek}
        />
      </div>
    </div>
  );
}
