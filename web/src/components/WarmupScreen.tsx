export function WarmupScreen() {
  return (
    <div className="warmup">
      <div className="warmup-glow" />
      <div className="warmup-content">
        <div className="warmup-rings">
          <div className="warmup-ring warmup-ring--outer" />
          <div className="warmup-ring warmup-ring--inner" />
          <div className="warmup-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
            </svg>
          </div>
        </div>
        <span className="warmup-text">Initializing GPU&hellip;</span>
        <span className="warmup-subtext">Compiling shaders for your device</span>
      </div>
    </div>
  );
}
