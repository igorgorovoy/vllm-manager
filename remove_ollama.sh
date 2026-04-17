#!/usr/bin/env bash
set -euo pipefail

echo "Stopping Ollama service/process if present..."
if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl stop ollama 2>/dev/null || true
  sudo systemctl disable ollama 2>/dev/null || true
fi
pkill -f "/usr/local/bin/ollama" 2>/dev/null || true

echo "Removing Ollama binary/service files..."
sudo rm -f /usr/local/bin/ollama
sudo rm -f /etc/systemd/system/ollama.service
sudo rm -f /etc/systemd/system/default.target.wants/ollama.service
if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl daemon-reload || true
fi

echo "Removing local Ollama data..."
rm -rf "$HOME/.ollama"

echo "Done. Ollama removed."
