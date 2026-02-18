.PHONY: wasm web dev clean

# Build the Rust STFT → WASM module into web/src/wasm/
wasm:
	wasm-pack build demucs-wasm --target web --out-dir ../web/src/wasm

# Install web dependencies (if needed) and build for production
web: wasm
	cd web && pnpm install && pnpm exec vite build

# Start Vite dev server (rebuilds WASM first)
dev: wasm
	cd web && pnpm install && pnpm exec vite

# Remove build artifacts
clean:
	cargo clean
	rm -rf web/src/wasm web/dist
