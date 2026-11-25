#!/usr/bin/env bash
set -e
export RETAILAI_PROJECT="${RETAILAI_PROJECT:-$HOME/Desktop/RetailAI}"
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8501 --reload
