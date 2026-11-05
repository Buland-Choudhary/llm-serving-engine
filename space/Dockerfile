# Hugging Face Space (Docker SDK, free CPU tier): OpenAI-compatible server + web console on :7860.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --chown=user pyproject.toml uv.lock README.md ./
COPY --chown=user src ./src
RUN uv sync --no-dev --frozen && chown -R user /app
USER user
ENV LLMSERVE_MODELS=/app/models
# Build the native kernels and fetch weights at image build time, not on the first request.
RUN uv run python -c "from llmserve.kernels import load_paged_attention as l; assert l()" \
 && uv run python -c "from llmserve.models.hub import resolve_model_path as r; r('smollm2-360m')"
EXPOSE 7860
CMD ["uv", "run", "llmserve", "serve", "--model", "smollm2-360m", "--host", "0.0.0.0", \
     "--port", "7860", "--threads", "2", "--max-num-seqs", "8", "--default-max-tokens", "256"]
