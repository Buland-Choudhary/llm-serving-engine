# CPU image for the OpenAI-compatible server.
#   docker build -t llmserve .
#   docker run -p 8000:8000 -v $PWD/models:/app/models llmserve --model smollm2-135m
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --no-dev --frozen && useradd -m app && chown -R app /app
USER app
# Build the native attention kernel at image build time, not on the first request.
RUN uv run python -c "from llmserve.kernels import load_paged_attention as l; assert l()"
EXPOSE 8000
ENTRYPOINT ["uv", "run", "llmserve", "serve", "--host", "0.0.0.0", "--port", "8000"]
