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
 * Manages synchronized playback of multiple audio stems.
 *
 * Each stem gets its own HTMLAudioElement. play/pause/seek are applied
 * to all elements simultaneously. Solo and mute use `audio.muted` so
 * playback stays in sync. A rAF loop corrects drift between elements.
 *
 * Solo and mute are mutually exclusive per track: soloing a track clears
 * its mute; muting a soloed track clears its solo.
 */
export function useMultiTrackPlayer(
  tracks: StemTrack[] | null,
): MultiTrackPlayer {
  const audioMapRef = useRef<Map<string, HTMLAudioElement>>(new Map());
  const rafRef = useRef(0);

  // Refs are the source of truth; React state mirrors them for rendering.
  const mutedRef = useRef<Record<string, boolean>>({});
  const soloRef = useRef<string | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [muted, setMuted] = useState<Record<string, boolean>>({});
  const [solo, setSolo] = useState<string | null>(null);

  /** Apply the correct muted state to every audio element. */
  const syncAudioMuted = useCallback(() => {
    const s = soloRef.current;
    const m = mutedRef.current;
    for (const [name, audio] of audioMapRef.current) {
      audio.muted = s !== null ? name !== s : (m[name] ?? false);
    }
  }, []);

  // Create / tear-down audio elements when tracks change
  useEffect(() => {
    const map = audioMapRef.current;

    // Cleanup old elements
    for (const audio of map.values()) {
      audio.pause();
      audio.src = "";
    }
    map.clear();

    if (!tracks || tracks.length === 0) {
      setIsPlaying(false);
      setCurrentTime(0);
      setDuration(0);
      setMuted({});
      setSolo(null);
      mutedRef.current = {};
      soloRef.current = null;
      return;
    }

    const initialMuted: Record<string, boolean> = {};
    for (const track of tracks) {
      const audio = new Audio(track.url);
      audio.preload = "auto";
      initialMuted[track.name] = false;
      map.set(track.name, audio);
    }
    mutedRef.current = initialMuted;
    soloRef.current = null;
    setMuted(initialMuted);
    setSolo(null);

    // Get duration from first track once loaded
    const first = map.values().next().value;
    if (first) {
      const onMeta = () => setDuration(first.duration);
      first.addEventListener("loadedmetadata", onMeta);

      const onEnded = () => setIsPlaying(false);
      first.addEventListener("ended", onEnded);
    }

    return () => {
      for (const audio of map.values()) {
        audio.pause();
        audio.src = "";
      }
      map.clear();
    };
  }, [tracks]);

  // rAF loop: update currentTime + correct drift between elements
  useEffect(() => {
    if (!isPlaying) return;

    const tick = () => {
      const entries = Array.from(audioMapRef.current.values());
      const leader = entries[0];
      if (leader) {
        const t = leader.currentTime;
        setCurrentTime(t);
        // Snap any element that has drifted >50 ms from the leader
        for (let i = 1; i < entries.length; i++) {
          if (Math.abs(entries[i].currentTime - t) > 0.05) {
            entries[i].currentTime = t;
          }
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(rafRef.current);
  }, [isPlaying]);

  const toggleAll = useCallback(() => {
    const map = audioMapRef.current;
    if (map.size === 0) return;

    const first = map.values().next().value!;
    if (first.paused) {
      // Sync all elements to the leader's time before starting
      const t = first.currentTime;
      for (const audio of map.values()) {
        audio.currentTime = t;
        audio.play();
      }
      setIsPlaying(true);
    } else {
      for (const audio of map.values()) audio.pause();
      setIsPlaying(false);
    }
  }, []);

  const playSolo = useCallback((name: string) => {
    const newSolo = soloRef.current === name ? null : name;
    soloRef.current = newSolo;
    setSolo(newSolo);

    // Solo & mute are exclusive per track: clear mute on the soloed track
    if (newSolo) {
      mutedRef.current = { ...mutedRef.current, [name]: false };
      setMuted({ ...mutedRef.current });
    }

    syncAudioMuted();

    // Auto-play when entering solo
    const first = audioMapRef.current.values().next().value;
    if (first?.paused) {
      const t = first.currentTime;
      for (const audio of audioMapRef.current.values()) {
        audio.currentTime = t;
        audio.play();
      }
      setIsPlaying(true);
    }
  }, [syncAudioMuted]);

  const toggleMute = useCallback((name: string) => {
    const wasMuted = mutedRef.current[name] ?? false;
    mutedRef.current = { ...mutedRef.current, [name]: !wasMuted };
    setMuted({ ...mutedRef.current });

    // Muting the currently-soloed track clears solo
    if (!wasMuted && soloRef.current === name) {
      soloRef.current = null;
      setSolo(null);
    }

    syncAudioMuted();
  }, [syncAudioMuted]);

  const seek = useCallback((time: number) => {
    for (const audio of audioMapRef.current.values()) {
      audio.currentTime = time;
    }
    setCurrentTime(time);
  }, []);

  return { isPlaying, currentTime, duration, muted, solo, toggleAll, playSolo, toggleMute, seek };
}
