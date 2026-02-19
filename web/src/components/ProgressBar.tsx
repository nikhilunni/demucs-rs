interface Props {
  /** 0–1 for determinate, null for indeterminate shimmer */
  progress: number | null;
}

export function ProgressBar({ progress }: Props) {
  const determinate = progress !== null;

  return (
    <div className="progress-track">
      <div
        className={`progress-fill ${determinate ? "" : "progress-fill--shimmer"}`}
        style={determinate ? { width: `${Math.round(progress * 100)}%` } : undefined}
      />
    </div>
  );
}
