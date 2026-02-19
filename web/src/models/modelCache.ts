/**
 * IndexedDB layer for caching downloaded model weights.
 *
 * DB: "demucs-models", store: "weights", version 1.
 * Keys are model IDs (e.g. "htdemucs").
 */

const DB_NAME = "demucs-models";
const STORE = "weights";
const DB_VERSION = 1;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** Check if a model is cached without reading the blob. */
export async function isCached(modelId: string): Promise<boolean> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const store = tx.objectStore(STORE);
    const req = store.count(modelId);
    req.onsuccess = () => resolve(req.result > 0);
    req.onerror = () => reject(req.error);
  });
}

/** Store model weights in IndexedDB. */
export async function cacheModel(
  modelId: string,
  data: ArrayBuffer,
): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const store = tx.objectStore(STORE);
    const req = store.put(data, modelId);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

/** Load cached model weights (returns null if not cached). */
export async function loadModel(
  modelId: string,
): Promise<ArrayBuffer | null> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const store = tx.objectStore(STORE);
    const req = store.get(modelId);
    req.onsuccess = () => resolve(req.result ?? null);
    req.onerror = () => reject(req.error);
  });
}
