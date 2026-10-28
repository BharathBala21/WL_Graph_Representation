#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null; done
