import { formatTime as fmt } from "../dsp/format";

interface Props {
  fileName: string;
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  onToggle: () => void;
}

export function PlayerControls({
  fileName,
  isPlaying,
  currentTime,
  duration,
  onToggle,
}: Props) {
  return (
    <div className="controls">
      <button
        className="play-btn"
        onClick={onToggle}
        title={isPlaying ? "Pause" : "Play"}
      >
        {isPlaying ? (
          <svg viewBox="0 0 24 24" fill="currentColor">
            <rect x="5" y="3" width="4" height="18" rx="1" />
            <rect x="15" y="3" width="4" height="18" rx="1" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="currentColor">
            <polygon points="6,3 20,12 6,21" />
          </svg>
        )}
      </button>

      <div className="track-info">
        <span className="track-name">{fileName}</span>
      </div>

      <div className="time-display">
        <span>{fmt(currentTime)}</span>
        <span className="sep">/</span>
        <span>{fmt(duration)}</span>
      </div>
    </div>
  );
}
