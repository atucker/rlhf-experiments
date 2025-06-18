#!/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11.9 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=cu124
uv pip install datasets==3.0.1
uv pip install -e .
uv pip install tokenizers==0.21.1 transformers==4.51.3