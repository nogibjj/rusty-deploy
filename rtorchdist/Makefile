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
	docker run -p 8080:8080 -e RUST_LOG="info,actix_web=info" -e RUST_BACKTRACE=1 pytorch-rust-distroless

test:
	RUST_BACKTRACE=1 RUST_LOG=debug,info cargo test -- --nocapture

make run:
	RUST_LOG="debug,info,actix_web=info"; \
	RUST_BACKTRACE=1; \
	cargo run
	
all: format lint test run