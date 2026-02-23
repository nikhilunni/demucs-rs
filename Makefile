.PHONY: cli wasm wasm-release web dev dev-release clean

# Auto-detect platform for native CLI build
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CLI_FEATURE := macos
else ifeq ($(UNAME_S),Linux)
  CLI_FEATURE := linux
else
  CLI_FEATURE := windows
endif

# Build native CLI (release, auto-detects GPU backend)
cli:
	cargo build -p demucs-cli --release --no-default-features --features $(CLI_FEATURE)

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

# Remove build artifacts
clean:
	cargo clean
	rm -rf web/src/wasm web/dist
