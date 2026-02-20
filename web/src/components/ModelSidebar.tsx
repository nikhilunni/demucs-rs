import { useEffect, useState } from "react";
import { getModels, type ModelVariant, type StemId, type SelectedModel } from "../models/registry";
import { useModelDownload, type DownloadState } from "../models/useModelDownload";
import { ProgressBar } from "./ProgressBar";

interface Props {
  onRun: (model: SelectedModel) => void;
}

function ModelCard({
  variant,
  selected,
  downloadState,
  onClick,
}: {
  variant: ModelVariant;
  selected: boolean;
  downloadState: DownloadState;
  onClick: () => void;
}) {
  const cached = downloadState.status === "cached";
  const downloading = downloadState.status === "downloading";

  return (
    <button
      className={`model-card ${selected ? "model-card--selected" : ""} ${cached && !selected ? "model-card--cached" : ""}`}
      onClick={onClick}
    >
      <div className="model-card__inner">
        <span className="model-card__label">{variant.label}</span>
        <span className="model-card__desc">{variant.description}</span>
        <span className="model-card__size">{variant.sizeMb} MB</span>

        <div className="model-card__status">
          {downloadState.status === "checking" && (
            <span className="model-card__hint">Checking...</span>
          )}
          {downloadState.status === "not-downloaded" && (
            <span className="model-card__hint">Click to download</span>
          )}
          {downloading && (
            <ProgressBar progress={(downloadState as Extract<DownloadState, { status: "downloading" }>).progress} />
          )}
          {cached && (
            <span className="model-card__cached">
              <svg className="model-card__check" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 6L9 17l-5-5" />
              </svg>
              Ready
            </span>
          )}
          {downloadState.status === "error" && (
            <span className="model-card__error">{downloadState.message}</span>
          )}
        </div>
      </div>
    </button>
  );
}

function StemChips({
  stems,
  selected,
  onToggle,
}: {
  stems: StemId[];
  selected: StemId[];
  onToggle: (stem: StemId) => void;
}) {
  return (
    <div className="stem-chips">
      <span className="stem-chips__label">Stems</span>
      <div className="stem-chips__list">
        {stems.map((stem) => (
          <button
            key={stem}
            className={`stem-chip ${selected.includes(stem) ? "stem-chip--active" : ""}`}
            onClick={() => onToggle(stem)}
          >
            {stem}
          </button>
        ))}
      </div>
    </div>
  );
}

export function ModelSidebar({ onRun }: Props) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedStems, setSelectedStems] = useState<StemId[]>([]);

  // Download hooks for each model
  const dl0 = useModelDownload(MODELS[0].id);
  const dl1 = useModelDownload(MODELS[1].id);
  const dl2 = useModelDownload(MODELS[2].id);
  const downloads = [dl0, dl1, dl2];

  const selectedVariant = MODELS.find((m) => m.id === selectedId) ?? null;
  const selectedIdx = MODELS.findIndex((m) => m.id === selectedId);
  const selectedDl = selectedIdx >= 0 ? downloads[selectedIdx] : null;

  // When model selection changes, reset stems to all
  useEffect(() => {
    if (selectedVariant) {
      setSelectedStems([...selectedVariant.stems]);
    }
  }, [selectedId]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleCardClick = (idx: number) => {
    const model = MODELS[idx];
    setSelectedId(model.id);

    // Auto-start download if not cached
    const dl = downloads[idx];
    if (dl.state.status === "not-downloaded" || dl.state.status === "error") {
      dl.start();
    }
  };

  const handleStemToggle = (stem: StemId) => {
    setSelectedStems((prev) => {
      // Don't allow deselecting the last stem
      if (prev.includes(stem) && prev.length === 1) return prev;
      return prev.includes(stem)
        ? prev.filter((s) => s !== stem)
        : [...prev, stem];
    });
  };

  const canRun =
    selectedVariant !== null &&
    selectedDl?.state.status === "cached" &&
    selectedStems.length > 0;

  return (
    <aside className="model-sidebar">
      <h3 className="model-sidebar__heading">Model</h3>

      <div className="model-sidebar__cards">
        {MODELS.map((variant, i) => (
          <ModelCard
            key={variant.id}
            variant={variant}
            selected={selectedId === variant.id}
            downloadState={downloads[i].state}
            onClick={() => handleCardClick(i)}
          />
        ))}
      </div>

      {selectedVariant?.id === "htdemucs_ft" && (
        <div className="model-sidebar__stems">
          <StemChips
            stems={selectedVariant.stems}
            selected={selectedStems}
            onToggle={handleStemToggle}
          />
        </div>
      )}

      <button
        className="run-btn"
        disabled={!canRun}
        onClick={() => {
          if (selectedVariant) {
            onRun({ variant: selectedVariant, stems: selectedStems });
          }
        }}
      >
        Run separation
      </button>
    </aside>
  );
}
