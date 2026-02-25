import { useCallback, useEffect, useRef, useState } from "react";

export interface StemTrack {
  name: string;
  url: string; // blob URL from encodeWavUrl
}

export interface MultiTrackPlayer {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  muted: Record<string, boolean>;
  solo: string | null;
  toggleAll: () => void;
  playSolo: (name: string) => void;
  toggleMute: (name: string) => void;
  seek: (time: number) => void;
}

/**
 * Manages synchronized playback of multiple audio stems using the Web Audio API.
 *
 * All stems are decoded into AudioBuffers and played through a single
 * AudioContext, guaranteeing sample-accurate sync. Mute and solo are
 * implemented via GainNodes (gain 0 or 1) so decoding is never interrupted.
 *
 * AudioBufferSourceNodes are one-shot, so pause/play recreates them at the
 * saved offset. The rAF loop simply reads the AudioContext clock.
 */
export function useMultiTrackPlayer(
  tracks: StemTrack[] | null,
): MultiTrackPlayer {
  // Web Audio API refs
  const ctxRef = useRef<AudioContext | null>(null);
  const buffersRef = useRef<Map<string, AudioBuffer>>(new Map());
  const gainsRef = useRef<Map<string, GainNode>>(new Map());
  const sourcesRef = useRef<Map<string, AudioBufferSourceNode>>(new Map());

  // Playback position tracking
  const startedAtRef = useRef(0); // audioCtx.currentTime when playback last started
  const offsetRef = useRef(0); // position in track (seconds) when playback last started
  const playingRef = useRef(false);

  const rafRef = useRef(0);

  // Mute/solo refs (source of truth)
  const mutedRef = useRef<Record<string, boolean>>({});
  const soloRef = useRef<string | null>(null);

  // React state for rendering
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [muted, setMuted] = useState<Record<string, boolean>>({});
  const [solo, setSolo] = useState<string | null>(null);

  /** Apply mute/solo by setting gain values. Instant, glitch-free. */
  const syncGains = useCallback(() => {
    const s = soloRef.current;
    const m = mutedRef.current;
    for (const [name, gain] of gainsRef.current) {
      const shouldMute = s !== null ? name !== s : (m[name] ?? false);
      gain.gain.value = shouldMute ? 0 : 1;
    }
  }, []);

  /** Stop all active source nodes, clearing onended handlers first. */
  const stopSources = useCallback(() => {
    for (const src of sourcesRef.current.values()) {
      src.onended = null;
      try {
        src.stop();
      } catch {
        /* already stopped */
      }
    }
    sourcesRef.current.clear();
  }, []);

  /** Create and start source nodes for all stems at the given offset. */
  const startSources = useCallback(
    (offset: number) => {
      const ctx = ctxRef.current;
      if (!ctx) return;

      stopSources();

      for (const [name, buffer] of buffersRef.current) {
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(gainsRef.current.get(name)!);
        source.start(0, offset);
        sourcesRef.current.set(name, source);
      }

      // Handle natural end of playback (only need one handler)
      const firstSource = sourcesRef.current.values().next().value;
      if (firstSource) {
        firstSource.onended = () => {
          if (playingRef.current) {
            playingRef.current = false;
            offsetRef.current = 0;
            setIsPlaying(false);
            setCurrentTime(0);
          }
        };
      }

      startedAtRef.current = ctx.currentTime;
      offsetRef.current = offset;
      playingRef.current = true;
    },
    [stopSources],
  );

  // Decode audio buffers when tracks change
  useEffect(() => {
    stopSources();
    buffersRef.current.clear();
    for (const gain of gainsRef.current.values()) gain.disconnect();
    gainsRef.current.clear();

    playingRef.current = false;
    offsetRef.current = 0;
    startedAtRef.current = 0;
    mutedRef.current = {};
    soloRef.current = null;

    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setMuted({});
    setSolo(null);

    if (!tracks || tracks.length === 0) return;

    // Create AudioContext on first use
    if (!ctxRef.current) {
      ctxRef.current = new AudioContext();
    }
    const ctx = ctxRef.current;

    let cancelled = false;

    (async () => {
      const initialMuted: Record<string, boolean> = {};

      // Decode all tracks in parallel
      const decoded = await Promise.all(
        tracks.map(async (track) => {
          const resp = await fetch(track.url);
          const arrayBuf = await resp.arrayBuffer();
          const audioBuf = await ctx.decodeAudioData(arrayBuf);
          return { name: track.name, buffer: audioBuf };
        }),
      );

      if (cancelled) return;

      for (const { name, buffer } of decoded) {
        buffersRef.current.set(name, buffer);

        // Create persistent gain node for each stem
        const gain = ctx.createGain();
        gain.connect(ctx.destination);
        gainsRef.current.set(name, gain);

        initialMuted[name] = false;
      }

      mutedRef.current = initialMuted;
      setMuted(initialMuted);

      // Duration from first buffer
      const firstBuf = buffersRef.current.values().next().value;
      if (firstBuf) setDuration(firstBuf.duration);
    })();

    return () => {
      cancelled = true;
    };
  }, [tracks, stopSources]);

  // rAF loop: read the shared AudioContext clock (no drift correction needed)
  useEffect(() => {
    if (!isPlaying) return;

    const tick = () => {
      const ctx = ctxRef.current;
      if (ctx && playingRef.current) {
        const t =
          offsetRef.current + (ctx.currentTime - startedAtRef.current);
        setCurrentTime(t);
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(rafRef.current);
  }, [isPlaying]);

  const toggleAll = useCallback(() => {
    const ctx = ctxRef.current;
    if (!ctx || buffersRef.current.size === 0) return;

    // Resume context if suspended (browser autoplay policy)
    if (ctx.state === "suspended") ctx.resume();

    if (playingRef.current) {
      // Pause: record position, stop sources
      const elapsed = ctx.currentTime - startedAtRef.current;
      offsetRef.current = offsetRef.current + elapsed;
      stopSources();
      playingRef.current = false;
      setIsPlaying(false);
    } else {
      // Play from saved offset
      startSources(offsetRef.current);
      syncGains();
      setIsPlaying(true);
    }
  }, [stopSources, startSources, syncGains]);

  const playSolo = useCallback(
    (name: string) => {
      const ctx = ctxRef.current;
      if (!ctx) return;
      if (ctx.state === "suspended") ctx.resume();

      const newSolo = soloRef.current === name ? null : name;
      soloRef.current = newSolo;
      setSolo(newSolo);

      // Solo & mute are exclusive per track: clear mute on the soloed track
      if (newSolo) {
        mutedRef.current = { ...mutedRef.current, [name]: false };
        setMuted({ ...mutedRef.current });
      }

      syncGains();

      // Auto-play when entering solo
      if (!playingRef.current) {
        startSources(offsetRef.current);
        syncGains();
        setIsPlaying(true);
      }
    },
    [syncGains, startSources],
  );

  const toggleMute = useCallback(
    (name: string) => {
      const wasMuted = mutedRef.current[name] ?? false;
      mutedRef.current = { ...mutedRef.current, [name]: !wasMuted };
      setMuted({ ...mutedRef.current });

      // Muting the currently-soloed track clears solo
      if (!wasMuted && soloRef.current === name) {
        soloRef.current = null;
        setSolo(null);
      }

      syncGains();
    },
    [syncGains],
  );

  const seek = useCallback(
    (time: number) => {
      if (playingRef.current) {
        // Stop and restart at new position
        startSources(time);
        syncGains();
      } else {
        offsetRef.current = time;
      }
      setCurrentTime(time);
    },
    [startSources, syncGains],
  );

  return {
    isPlaying,
    currentTime,
    duration,
    muted,
    solo,
    toggleAll,
    playSolo,
    toggleMute,
    seek,
  };
}
