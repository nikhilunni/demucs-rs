import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { spectrogramDisplay } from "../design/tokens";

function fmt(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

interface Props {
  image: HTMLCanvasElement;
  currentTime: number;
  duration: number;
  sampleRate: number;
  onSeek: (time: number) => void;
  clipRange?: [number, number] | null;
  onClipChange?: (range: [number, number] | null) => void;
}

type DragAction = "create" | "resize-left" | "resize-right" | "move";

const DRAG_THRESHOLD = 5; // px to distinguish click from drag
const MIN_SELECTION = 0.5; // seconds

export function SpectrogramView({
  image,
  currentTime,
  duration,
  sampleRate,
  onSeek,
  clipRange = null,
  onClipChange,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  // One-time tooltip state
  const [showTooltip, setShowTooltip] = useState(false);
  const tooltipTimerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Local preview range for smooth dragging
  const [previewRange, setPreviewRange] = useState<[number, number] | null>(null);
  const activeRange = previewRange ?? clipRange;

  // Refs for drag state (avoid re-renders during drag)
  const dragRef = useRef<{
    action: DragAction;
    startX: number;
    startTime: number;
    origRange: [number, number] | null;
  } | null>(null);

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

  /* ── Helpers: pixel <-> time ── */
  const pxToTime = useCallback(
    (clientX: number) => {
      const rect = wrapRef.current?.getBoundingClientRect();
      if (!rect || !duration) return 0;
      const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      return pct * duration;
    },
    [duration],
  );

  const timeToPct = useCallback(
    (time: number) => {
      if (!duration) return 0;
      return (time / duration) * 100;
    },
    [duration],
  );

  /* ── Hit-test: which drag action based on click position ── */
  const hitTest = useCallback(
    (clientX: number): { action: DragAction; range: [number, number] | null } => {
      if (!activeRange || !wrapRef.current) {
        return { action: "create", range: null };
      }

      const rect = wrapRef.current.getBoundingClientRect();
      const leftPx = (activeRange[0] / duration) * rect.width + rect.left;
      const rightPx = (activeRange[1] / duration) * rect.width + rect.left;
      const HANDLE_ZONE = 10; // px

      if (Math.abs(clientX - leftPx) <= HANDLE_ZONE) {
        return { action: "resize-left", range: activeRange };
      }
      if (Math.abs(clientX - rightPx) <= HANDLE_ZONE) {
        return { action: "resize-right", range: activeRange };
      }
      if (clientX > leftPx && clientX < rightPx) {
        return { action: "move", range: activeRange };
      }
      return { action: "create", range: null };
    },
    [activeRange, duration],
  );

  /* ── Mouse handling ── */
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!onClipChange || !duration) return;
      if (e.button !== 0) return; // left click only

      const { action, range } = hitTest(e.clientX);
      const time = pxToTime(e.clientX);

      dragRef.current = {
        action,
        startX: e.clientX,
        startTime: time,
        origRange: range ? [...range] : null,
      };

      const handleMouseMove = (me: MouseEvent) => {
        const drag = dragRef.current;
        if (!drag) return;

        const dx = me.clientX - drag.startX;

        // Haven't passed threshold yet — don't start dragging
        if (Math.abs(dx) < DRAG_THRESHOLD && drag.action === "create" && !drag.origRange) {
          return;
        }

        const t = pxToTime(me.clientX);

        let newRange: [number, number];

        switch (drag.action) {
          case "create": {
            const a = drag.startTime;
            const b = t;
            newRange = a < b ? [a, b] : [b, a];
            break;
          }
          case "resize-left": {
            newRange = [Math.min(t, drag.origRange![1] - MIN_SELECTION), drag.origRange![1]];
            break;
          }
          case "resize-right": {
            newRange = [drag.origRange![0], Math.max(t, drag.origRange![0] + MIN_SELECTION)];
            break;
          }
          case "move": {
            const orig = drag.origRange!;
            const len = orig[1] - orig[0];
            const dt = t - drag.startTime;
            let s = orig[0] + dt;
            let end = orig[1] + dt;
            // Clamp to bounds
            if (s < 0) {
              s = 0;
              end = len;
            }
            if (end > duration) {
              end = duration;
              s = duration - len;
            }
            newRange = [s, end];
            break;
          }
        }

        // Enforce min selection
        if (newRange[1] - newRange[0] < MIN_SELECTION) {
          return;
        }

        // Clamp to [0, duration]
        newRange[0] = Math.max(0, newRange[0]);
        newRange[1] = Math.min(duration, newRange[1]);

        setPreviewRange(newRange);
      };

      const handleMouseUp = (me: MouseEvent) => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);

        const drag = dragRef.current;
        dragRef.current = null;

        if (!drag) return;

        const dx = Math.abs(me.clientX - drag.startX);

        // Click (not drag) → seek
        if (dx < DRAG_THRESHOLD) {
          setPreviewRange(null);
          const time = pxToTime(me.clientX);
          onSeek(time);
          return;
        }

        // Commit the preview range
        setPreviewRange((prev) => {
          if (prev) onClipChange(prev);
          return null;
        });
      };

      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
    },
    [onClipChange, duration, hitTest, pxToTime, onSeek],
  );

  /* ── Dynamic cursor style ── */
  const handleMouseMoveForCursor = useCallback(
    (e: React.MouseEvent) => {
      if (!onClipChange || !activeRange || !wrapRef.current || dragRef.current) return;

      const rect = wrapRef.current.getBoundingClientRect();
      const leftPx = (activeRange[0] / duration) * rect.width + rect.left;
      const rightPx = (activeRange[1] / duration) * rect.width + rect.left;
      const HANDLE_ZONE = 10;

      if (
        Math.abs(e.clientX - leftPx) <= HANDLE_ZONE ||
        Math.abs(e.clientX - rightPx) <= HANDLE_ZONE
      ) {
        wrapRef.current.style.cursor = "ew-resize";
      } else if (e.clientX > leftPx && e.clientX < rightPx) {
        wrapRef.current.style.cursor = "grab";
      } else {
        wrapRef.current.style.cursor = "crosshair";
      }
    },
    [onClipChange, activeRange, duration],
  );

  const handleMouseLeave = useCallback(() => {
    if (wrapRef.current && !dragRef.current) {
      wrapRef.current.style.cursor = "crosshair";
    }
  }, []);

  /* ── One-time tooltip on first hover ── */
  const tooltipShownRef = useRef(false);

  const handleMouseEnter = useCallback(() => {
    if (!onClipChange || tooltipShownRef.current) return;
    tooltipShownRef.current = true;

    setShowTooltip(true);

    // Auto-dismiss after the CSS animation finishes (2.5s delay + 0.4s fade)
    clearTimeout(tooltipTimerRef.current);
    tooltipTimerRef.current = setTimeout(() => setShowTooltip(false), 3000);
  }, [onClipChange]);

  // Also dismiss tooltip on first drag
  useEffect(() => {
    if (activeRange && showTooltip) {
      setShowTooltip(false);
      clearTimeout(tooltipTimerRef.current);
    }
  }, [activeRange, showTooltip]);

  useEffect(() => () => clearTimeout(tooltipTimerRef.current), []);

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

  // Clip overlay percentages
  const clipEnabled = onClipChange != null;
  const leftPct = activeRange ? timeToPct(activeRange[0]) : 0;
  const rightPct = activeRange ? timeToPct(activeRange[1]) : 0;

  return (
    <div
      ref={wrapRef}
      className="spectrogram-wrap"
      onMouseDown={clipEnabled ? handleMouseDown : undefined}
      onClick={clipEnabled ? undefined : (e) => {
        const rect = wrapRef.current?.getBoundingClientRect();
        if (!rect || !duration) return;
        const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        onSeek(pct * duration);
      }}
      onMouseEnter={clipEnabled ? handleMouseEnter : undefined}
      onMouseMove={clipEnabled ? handleMouseMoveForCursor : undefined}
      onMouseLeave={clipEnabled ? handleMouseLeave : undefined}
    >
      <canvas ref={canvasRef} className="spectrogram-canvas" />
      <div className="cursor" style={{ left: `${cursorPct}%` }} />

      {/* Clip selection overlay */}
      {activeRange && clipEnabled && (
        <>
          {/* Dim areas outside selection */}
          <div className="clip-dim" style={{ left: 0, width: `${leftPct}%` }} />
          <div className="clip-dim" style={{ right: 0, width: `${100 - rightPct}%` }} />

          {/* Selection border */}
          <div
            className="clip-selection"
            style={{ left: `${leftPct}%`, width: `${rightPct - leftPct}%` }}
          />

          {/* Left handle */}
          <div className="clip-handle" style={{ left: `${leftPct}%` }}>
            <div className="clip-handle__grip" />
          </div>
          <div className="clip-time" style={{ left: `${leftPct}%` }}>
            {fmt(activeRange[0])}
          </div>

          {/* Right handle */}
          <div className="clip-handle" style={{ left: `${rightPct}%` }}>
            <div className="clip-handle__grip" />
          </div>
          <div className="clip-time" style={{ left: `${rightPct}%` }}>
            {fmt(activeRange[1])}
          </div>

          {/* Clear button */}
          <div
            className="clip-clear"
            style={{
              left: `${(leftPct + rightPct) / 2}%`,
            }}
            onMouseDown={(e) => {
              e.stopPropagation();
              onClipChange!(null);
            }}
          >
            ×
          </div>
        </>
      )}

      {showTooltip && !activeRange && (
        <div className="clip-tooltip">drag to select a region</div>
      )}

      {freqLabels.map(({ freq, pct, label }) => (
        <span key={freq} className="freq-label" style={{ top: `${pct}%` }}>
          {label}
        </span>
      ))}
    </div>
  );
}
