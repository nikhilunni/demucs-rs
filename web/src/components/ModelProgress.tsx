import type { ProgressEvent } from "../hooks/useDemucsWorker";

// ─── State ──────────────────────────────────────────────────────────────────

export interface ModelProgressState {
  chunk: { index: number; total: number } | null;
  /** Number of completed layers per domain (0–4) */
  freqEnc: number;
  timeEnc: number;
  transformer: boolean;
  freqDec: number;
  timeDec: number;
  /** Currently active stage identifier */
  active:
    | "freq_enc"
    | "time_enc"
    | "transformer"
    | "freq_dec"
    | "time_dec"
    | "stems"
    | null;
}

export const INITIAL_PROGRESS: ModelProgressState = {
  chunk: null,
  freqEnc: 0,
  timeEnc: 0,
  transformer: false,
  freqDec: 0,
  timeDec: 0,
  active: null,
};

/** Reduce a progress event into the next state. */
export function reduceProgress(
  prev: ModelProgressState,
  event: ProgressEvent,
): ModelProgressState {
  switch (event.type) {
    case "chunk_started":
      return {
        ...INITIAL_PROGRESS,
        chunk: { index: event.index, total: event.total },
        active: "freq_enc",
      };

    case "encoder_done":
      if (event.domain === "freq") {
        const allFreqDone = event.layer + 1 >= event.numLayers;
        return {
          ...prev,
          freqEnc: event.layer + 1,
          active: allFreqDone ? "time_enc" : "freq_enc",
        };
      } else {
        const allTimeDone = event.layer + 1 >= event.numLayers;
        return {
          ...prev,
          timeEnc: event.layer + 1,
          active: allTimeDone ? "transformer" : "time_enc",
        };
      }

    case "transformer_done":
      return { ...prev, transformer: true, active: "freq_dec" };

    case "decoder_done":
      if (event.domain === "freq") {
        const allFreqDone = event.layer + 1 >= event.numLayers;
        return {
          ...prev,
          freqDec: event.layer + 1,
          active: allFreqDone ? "time_dec" : "freq_dec",
        };
      } else {
        const allTimeDone = event.layer + 1 >= event.numLayers;
        return {
          ...prev,
          timeDec: event.layer + 1,
          active: allTimeDone ? "stems" : "time_dec",
        };
      }

    case "denormalized":
      return { ...prev, active: "stems" };

    case "stem_done":
      return prev;

    case "chunk_done":
      return {
        ...prev,
        chunk: { index: event.index, total: event.total },
        active: null,
      };

    default:
      return prev;
  }
}

// ─── Component ──────────────────────────────────────────────────────────────

const NUM_LAYERS = 4;

type NodeStatus = "done" | "active" | "pending";

interface Props {
  progress: ModelProgressState;
}

export function ModelProgress({ progress }: Props) {
  const {
    chunk,
    freqEnc,
    timeEnc,
    transformer,
    freqDec,
    timeDec,
    active,
  } = progress;

  // Chunk progress
  const chunkTotal = chunk?.total ?? 1;
  const chunksDone = chunk ? chunk.index + (active === null ? 1 : 0) : 0;
  const chunkPct = Math.round((chunksDone / chunkTotal) * 100);
  const isChunked = chunkTotal > 1;

  // Node status helpers
  const encStatus = (domain: "freq" | "time", layer: number): NodeStatus => {
    const done = domain === "freq" ? freqEnc : timeEnc;
    const isActive = domain === "freq" ? active === "freq_enc" : active === "time_enc";
    if (layer < done) return "done";
    if (layer === done && isActive) return "active";
    return "pending";
  };

  const xfmrStatus: NodeStatus = transformer
    ? "done"
    : active === "transformer"
      ? "active"
      : "pending";

  const decStatus = (domain: "freq" | "time", layer: number): NodeStatus => {
    const done = domain === "freq" ? freqDec : timeDec;
    const isActive = domain === "freq" ? active === "freq_dec" : active === "time_dec";
    if (layer < done) return "done";
    if (layer === done && isActive) return "active";
    return "pending";
  };

  // Stage label
  const stageLabel =
    active === "stems"
      ? "Extracting stems"
      : active != null
        ? "Processing with WebGPU\u2026"
        : chunksDone > 0
          ? "Waiting..."
          : "Starting...";

  // Completed stages for progress bar
  const doneStages = freqEnc + timeEnc + (transformer ? 1 : 0) + freqDec + timeDec;
  const totalStages = NUM_LAYERS * 4 + 1; // 4 domains × 4 layers + transformer
  const stagePct = Math.round((doneStages / totalStages) * 100);

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

      {/* Architecture diagram */}
      <div className="mp-arch">
        {/* Header labels */}
        <div className="mp-arch__header">
          <span className="mp-arch__title mp-arch__title--freq">Frequency</span>
          <span className="mp-arch__title mp-arch__title--time">Temporal</span>
        </div>

        {/* Encoder section */}
        <div className="mp-section">
          <div className="mp-section__label">Encoder</div>
          <div className="mp-tracks">
            <Track domain="freq" layers={NUM_LAYERS} statusFn={(i) => encStatus("freq", i)} />
            <Track domain="time" layers={NUM_LAYERS} statusFn={(i) => encStatus("time", i)} />
          </div>
        </div>

        {/* Transformer */}
        <div className="mp-section mp-section--xfmr">
          <div className="mp-section__label"></div>
          <div className="mp-xfmr">
            <div className={`mp-node mp-node--xfmr mp-node--${xfmrStatus}`}>
              <div className="mp-node__inner" />
              <span className="mp-node__label">Cross-Attention</span>
            </div>
          </div>
        </div>

        {/* Decoder section */}
        <div className="mp-section">
          <div className="mp-section__label">Decoder</div>
          <div className="mp-tracks">
            <Track domain="freq" layers={NUM_LAYERS} statusFn={(i) => decStatus("freq", i)} />
            <Track domain="time" layers={NUM_LAYERS} statusFn={(i) => decStatus("time", i)} />
          </div>
        </div>
      </div>

      {/* Detail line */}
      <div className="mp-detail">
        <span className="mp-detail__stage">{stageLabel}</span>
        <span className="mp-detail__count">{doneStages}/{totalStages}</span>
      </div>

      {/* Overall progress bar */}
      <div className="mp-bar">
        <div className="mp-bar__fill" style={{ width: `${stagePct}%` }} />
      </div>
    </div>
  );
}

// ─── Track (row of nodes for one domain) ────────────────────────────────────

function Track({
  domain,
  layers,
  statusFn,
}: {
  domain: "freq" | "time";
  layers: number;
  statusFn: (i: number) => NodeStatus;
}) {
  return (
    <div className={`mp-track mp-track--${domain}`}>
      {Array.from({ length: layers }, (_, i) => {
        const status = statusFn(i);
        return (
          <div key={i} className={`mp-node mp-node--${status} mp-node--${domain}`}>
            <div className="mp-node__inner" />
          </div>
        );
      })}
    </div>
  );
}
