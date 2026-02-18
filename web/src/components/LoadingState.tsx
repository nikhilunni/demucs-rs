export function LoadingState() {
  return (
    <div className="loading">
      <div className="loading-bars">
        {Array.from({ length: 5 }, (_, i) => (
          <div
            key={i}
            className="loading-bar"
            style={{ animationDelay: `${i * 0.12}s` }}
          />
        ))}
      </div>
      <span className="loading-text">Computing spectrogram&hellip;</span>
    </div>
  );
}
