import { useCallback, useEffect, useMemo, useRef } from "react";
import { spectrogramDisplay } from "../design/tokens";

interface Props {
  image: HTMLCanvasElement;
  currentTime: number;
  duration: number;
  sampleRate: number;
  onSeek: (time: number) => void;
}

export function SpectrogramView({
  image,
  currentTime,
  duration,
  sampleRate,
  onSeek,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  /* ── Draw the spectrogram image scaled to the display canvas ── */
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap || !image) return;

    const rect = wrap.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext("2d")!;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  }, [image]);

  useEffect(() => {
    draw();
    if (!wrapRef.current) return;
    const obs = new ResizeObserver(draw);
    obs.observe(wrapRef.current);
    return () => obs.disconnect();
  }, [draw]);

  /* ── Click-to-seek ── */
  const handleClick = (e: React.MouseEvent) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect || !duration) return;
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(pct * duration);
  };

  /* ── Frequency axis labels ── */
  const freqLabels = useMemo(() => {
    const nyquist = sampleRate / 2;
    const logMin = Math.log10(spectrogramDisplay.minFreq);
    const logMax = Math.log10(nyquist);
    const logRange = logMax - logMin;

    return spectrogramDisplay.freqTicks
      .filter((f) => f >= spectrogramDisplay.minFreq && f <= nyquist)
      .map((f) => ({
        freq: f,
        pct: ((logMax - Math.log10(f)) / logRange) * 100,
        label: f >= 1000 ? `${f / 1000}k` : `${f}`,
      }));
  }, [sampleRate]);

  const cursorPct = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div ref={wrapRef} className="spectrogram-wrap" onClick={handleClick}>
      <canvas ref={canvasRef} className="spectrogram-canvas" />
      <div className="cursor" style={{ left: `${cursorPct}%` }} />
      {freqLabels.map(({ freq, pct, label }) => (
        <span key={freq} className="freq-label" style={{ top: `${pct}%` }}>
          {label}
        </span>
      ))}
    </div>
  );
}
