format:
	cargo fmt --quiet

lint:
	cargo clippy --quiet

refactor: format lint

build:
	cargo build -j 16 --release

dbuild:
	docker build -t pytorch-rust-distroless .

drun:
	docker run -p 8080:8080 pytorch-rust-distroless

test:
	RUST_LOG=debug,info cargo test -- --nocapture
.PHONY: run

run:
	#cargo build -j 15 --release && \
	export RUST_BACKTRACE=1 && \
	export RUSTFLAGS="-C target-cpu=native" && \
	export XNNPACK_DISABLE=1 && \
	cargo run

all: format lint test run