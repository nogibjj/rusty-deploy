format:
	cargo fmt --quiet

lint:
	cargo clippy --quiet

build:
	cargo build -j 6

dbuild:
	docker build -t pytorch-rust-distroless .

drun:
	docker run -p 8080:8080 pytorch-rust-distroless

test:
	cargo test --quiet

.PHONY: run

run:
	export RUSTFLAGS="-C target-cpu=native" && \
	export XNNPACK_DISABLE=1 && \
	cargo run

all: format lint test run