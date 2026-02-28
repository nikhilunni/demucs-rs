import { test, expect } from "../fixtures/app";

test.describe("Model download + cache", () => {
  test("downloads model and persists in IndexedDB across reloads", async ({
    demucsApp,
  }) => {
    await demucsApp.goto();
    await demucsApp.waitForWarmup();

    // Load audio first (model sidebar appears after audio is loaded)
    await demucsApp.loadAudioFile();
    await demucsApp.waitForSpectrogram();

    // Click the first model card (htdemucs, 84 MB)
    await demucsApp.selectModel(0);

    // Wait for download to complete — "Ready" badge visible
    await demucsApp.waitForModelCached(0);

    // Verify model exists in IndexedDB
    const cached = await demucsApp.page.evaluate(async () => {
      return new Promise<boolean>((resolve) => {
        const req = indexedDB.open("demucs-models");
        req.onsuccess = () => {
          const db = req.result;
          const stores = Array.from(db.objectStoreNames);
          if (stores.length === 0) {
            db.close();
            resolve(false);
            return;
          }
          const tx = db.transaction(stores[0], "readonly");
          const store = tx.objectStore(stores[0]);
          const getReq = store.get("htdemucs");
          getReq.onsuccess = () => {
            db.close();
            resolve(getReq.result != null);
          };
          getReq.onerror = () => {
            db.close();
            resolve(false);
          };
        };
        req.onerror = () => resolve(false);
      });
    });
    expect(cached).toBe(true);

    // Reload and verify cache persists — "Ready" badge should appear immediately
    await demucsApp.page.reload();
    await demucsApp.waitForWarmup();
    await demucsApp.loadAudioFile();
    await demucsApp.waitForSpectrogram();

    // The "Ready" badge should be visible without re-downloading
    await expect(
      demucsApp.page
        .locator(".model-card")
        .nth(0)
        .locator(".model-card__cached"),
    ).toBeVisible({ timeout: 10_000 });
  });
});
