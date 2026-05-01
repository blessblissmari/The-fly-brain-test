# Multi-stage build: Rust core compiled inside the image, then a slim runtime.
#
# Stage 1: build the maturin wheel.
# Stage 2: install runtime deps and the wheel into a Python slim image.

ARG PYTHON_VERSION=3.11
ARG RUST_VERSION=1.83

FROM rust:${RUST_VERSION}-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl pkg-config libssl-dev \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip python${PYTHON_VERSION}-venv \
 && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m pip install --upgrade pip "maturin>=1.7,<2"

WORKDIR /build
COPY . .

RUN maturin build --release --strip --interpreter python${PYTHON_VERSION}


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /build/target/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl
COPY pyproject.toml ./pyproject.toml
RUN pip install --no-deps . \
 && pip install \
        "pydantic>=2.6,<3" \
        "hydra-core>=1.3,<2" \
        "omegaconf>=2.3,<3" \
        "yandex-cloud-ml-sdk>=0.4,<1" \
        "tenacity>=8.2,<9" \
        "tqdm>=4.66" \
        "rich>=13.7" \
        "numpy>=1.26,<3"

COPY flybrain ./flybrain
COPY configs ./configs
COPY scripts ./scripts

ENV PYTHONPATH=/workspace
ENTRYPOINT ["python", "-m", "flybrain.cli"]
CMD ["info"]
