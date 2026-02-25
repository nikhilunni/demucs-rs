import { useEffect, useState } from "react";
import type { ProgressEvent } from "../hooks/useDemucsWorker";

// ─── State ──────────────────────────────────────────────────────────────────

export interface ModelProgressState {
  chunk: { index: number; total: number } | null;
  done: boolean;
}

export const INITIAL_PROGRESS: ModelProgressState = {
  chunk: null,
  done: false,
};

/** Reduce a progress event into the next state. */
export function reduceProgress(
  prev: ModelProgressState,
  event: ProgressEvent,
): ModelProgressState {
  switch (event.type) {
    case "chunk_started":
      return { chunk: { index: event.index, total: event.total }, done: false };

    case "chunk_done":
      return { chunk: { index: event.index, total: event.total }, done: true };

    default:
      return prev;
  }
}

// ─── Component ──────────────────────────────────────────────────────────────

interface Props {
  progress: ModelProgressState;
}

export function ModelProgress({ progress }: Props) {
  const { chunk } = progress;

  // Elapsed timer
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const chunkTotal = chunk?.total ?? 1;
  const isChunked = chunkTotal > 1;
  const chunksDone = chunk ? chunk.index + (progress.done ? 1 : 0) : 0;
  const chunkPct = Math.round((chunksDone / chunkTotal) * 100);

  return (
    <div className="mp">
      {/* Chunk progress */}
      {isChunked && (
        <div className="mp-chunk">
          <span className="mp-chunk__label">
            Chunk {(chunk?.index ?? 0) + 1} of {chunkTotal}
          </span>
          <div className="mp-chunk__bar">
            <div className="mp-chunk__fill" style={{ width: `${chunkPct}%` }} />
          </div>
        </div>
      )}

      {/* Detail line */}
      <div className="mp-detail">
        <span className="mp-detail__stage">Processing with WebGPU&hellip; {elapsed}s</span>
      </div>

      {/* Static accent bar */}
      <div className="mp-bar">
        <div className="mp-bar__fill" style={{ width: "100%" }} />
      </div>
    </div>
  );
}
