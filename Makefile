.PHONY: cli plugin wasm wasm-release web dev dev-release clean test-plugin test-plugin-e2e test-web-e2e test-web-smoke

# Build native CLI (release, auto-detects GPU backend via target-conditional deps)
cli:
	cargo build -p demucs-cli --release

# Build and bundle DAW plugin (VST3 + CLAP)
plugin:
	cargo xtask bundle demucs-plugin --release

# Build WASM (debug — fast compile, slow runtime)
wasm:
	wasm-pack build demucs-wasm --target web --out-dir ../web/src/wasm

# Build WASM (release — slow compile, fast runtime)
wasm-release:
	wasm-pack build demucs-wasm --release --target web --out-dir ../web/src/wasm

# Install web dependencies (if needed) and build for production
web: wasm-release
	cd web && pnpm install && pnpm exec vite build

# Start Vite dev server (debug WASM — fast iteration)
dev: wasm
	cd web && pnpm install && pnpm exec vite

# Start Vite dev server (release WASM — for performance testing)
dev-release: wasm-release
	cd web && pnpm install && pnpm exec vite

# ── Tests ──────────────────────────────────────────────────────────────────────

# Plugin unit tests (audio mixing, MIDI gating, duration conversion)
test-plugin:
	cargo test -p demucs-plugin

# Full E2E inference test (requires GPU + cached model weights)
test-plugin-e2e:
	cargo test -p demucs-core --test stem_sum -- --ignored

# Web app E2E tests (requires production build + Chrome)
test-web-e2e:
	cd web && pnpm exec playwright test --config e2e/playwright.config.ts

# Web smoke test only (load audio + spectrogram render)
test-web-smoke:
	cd web && pnpm exec playwright test --config e2e/playwright.config.ts --grep @smoke

# Remove build artifacts
clean:
	cargo clean
	rm -rf web/src/wasm web/dist
