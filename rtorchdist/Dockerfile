# First, we need a Rust build environment with Libtorch (PyTorch C++ backend) installed
FROM rust:1.67.1 as build-env
WORKDIR /app

# Install Libtorch (PyTorch C++ backend) v2.0.0
RUN curl -LO https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip && \
    mv libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip

# Set LIBTORCH environment variable
ENV LIBTORCH=/app/libtorch

# Copy the Rust code, the pre-trained PyTorch model, and the resnet34.ot file
COPY . /app

# Build the Rust Actix-web application with LIBTORCH set
ENV MODEL_PATH="/model/resnet34.ot"
RUN LIBTORCH=${LIBTORCH} cargo build --release

# Use a distroless cc image
FROM gcr.io/distroless/cc
COPY --from=build-env /app/target/release/rtorchdist /
COPY --from=build-env /app/model /model
COPY --from=build-env /app/libtorch /app/libtorch
COPY --from=build-env /app/model/resnet34.ot /resnet34.ot
COPY --from=build-env /app/tests/fixtures/lion.jpg /tests/fixtures/lion.jpg
ENV IMAGE_PATH /tests/fixtures/lion.jpg


# Set the required environment variables for Libtorch
ENV LD_LIBRARY_PATH="/app/libtorch/lib:${LD_LIBRARY_PATH}"
# Your container will fail if you haven't downloaded the model and set the path
ENV MODEL_PATH="/model/resnet34.ot"
# Start the Actix-web server
CMD ["./rtorchdist"]
