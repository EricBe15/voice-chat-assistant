#!/bin/sh
# Sets everything up without starting the assistant: .venv, packages, models, llama.cpp.
# start.sh does the same on its first run, so this is optional.
exec "$(dirname "$0")/start.sh" --install-only "$@"
