#!/bin/bash
ollama serve &
python main.py
systemctl stop ollama.service