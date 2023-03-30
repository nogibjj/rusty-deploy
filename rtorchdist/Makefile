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

run:
	cargo run 

all: format lint test run