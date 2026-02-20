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
 * to all elements simultaneously. Solo and mute are implemented via
 * `audio.muted` so time stays in sync.
 */
export function useMultiTrackPlayer(
  tracks: StemTrack[] | null,
): MultiTrackPlayer {
  const audioMapRef = useRef<Map<string, HTMLAudioElement>>(new Map());
  const rafRef = useRef(0);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [muted, setMuted] = useState<Record<string, boolean>>({});
  const [solo, setSolo] = useState<string | null>(null);

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
      return;
    }

    const initialMuted: Record<string, boolean> = {};
    for (const track of tracks) {
      const audio = new Audio(track.url);
      audio.preload = "auto";
      initialMuted[track.name] = false;
      map.set(track.name, audio);
    }
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

  // rAF loop for currentTime updates
  useEffect(() => {
    if (!isPlaying) return;

    const tick = () => {
      const first = audioMapRef.current.values().next().value;
      if (first) setCurrentTime(first.currentTime);
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
      for (const audio of map.values()) audio.play();
      setIsPlaying(true);
    } else {
      for (const audio of map.values()) audio.pause();
      setIsPlaying(false);
    }
  }, []);

  const playSolo = useCallback((name: string) => {
    const map = audioMapRef.current;
    setSolo((prev) => {
      const newSolo = prev === name ? null : name;
      // Apply mute state based on solo
      for (const [n, audio] of map) {
        audio.muted = newSolo !== null && n !== newSolo;
      }
      return newSolo;
    });

    // Start playing if not already
    const first = map.values().next().value;
    if (first?.paused) {
      for (const audio of map.values()) audio.play();
      setIsPlaying(true);
    }
  }, []);

  const toggleMute = useCallback((name: string) => {
    setSolo(null); // clear solo when manually toggling mute
    setMuted((prev) => {
      const next = { ...prev, [name]: !prev[name] };
      const audio = audioMapRef.current.get(name);
      if (audio) audio.muted = next[name];
      // Also clear solo muting on all other tracks
      for (const [n, a] of audioMapRef.current) {
        if (n !== name) a.muted = next[n] ?? false;
      }
      return next;
    });
  }, []);

  const seek = useCallback((time: number) => {
    for (const audio of audioMapRef.current.values()) {
      audio.currentTime = time;
    }
    setCurrentTime(time);
  }, []);

  return { isPlaying, currentTime, duration, muted, solo, toggleAll, playSolo, toggleMute, seek };
}
