import { useCallback, useEffect, useRef, useState } from "react";
import { MODELS, modelUrl } from "./registry";
import { cacheModel, isCached } from "./modelCache";

export type DownloadState =
  | { status: "checking" }
  | { status: "not-downloaded" }
  | { status: "downloading"; progress: number | null }
  | { status: "cached" }
  | { status: "error"; message: string };

export function useModelDownload(modelId: string): {
  state: DownloadState;
  start: () => void;
} {
  const [state, setState] = useState<DownloadState>({ status: "checking" });
  const abortRef = useRef<AbortController | null>(null);

  // Check cache on mount / modelId change
  useEffect(() => {
    let cancelled = false;
    setState({ status: "checking" });

    isCached(modelId).then((cached) => {
      if (!cancelled) {
        setState(cached ? { status: "cached" } : { status: "not-downloaded" });
      }
    });

    return () => {
      cancelled = true;
      abortRef.current?.abort();
      abortRef.current = null;
    };
  }, [modelId]);

  const start = useCallback(() => {
    const variant = MODELS.find((m) => m.id === modelId);
    if (!variant) return;

    const controller = new AbortController();
    abortRef.current = controller;
    setState({ status: "downloading", progress: 0 });

    (async () => {
      try {
        const res = await fetch(modelUrl(variant), {
          signal: controller.signal,
        });
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }

        const contentLength = res.headers.get("content-length");
        const total = contentLength ? parseInt(contentLength, 10) : null;
        const reader = res.body!.getReader();
        const chunks: Uint8Array[] = [];
        let received = 0;

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          received += value.length;
          setState({
            status: "downloading",
            progress: total ? received / total : null,
          });
        }

        // Concatenate chunks into a single ArrayBuffer
        const buf = new Uint8Array(received);
        let offset = 0;
        for (const chunk of chunks) {
          buf.set(chunk, offset);
          offset += chunk.length;
        }

        await cacheModel(modelId, buf.buffer);
        setState({ status: "cached" });
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setState({
          status: "error",
          message: err instanceof Error ? err.message : "Download failed",
        });
      }
    })();
  }, [modelId]);

  return { state, start };
}
