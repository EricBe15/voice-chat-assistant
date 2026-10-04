#!/bin/sh
# Starts the assistant. The first run sets everything up (.venv, packages, models, llama.cpp).
#   ./start.sh            the assistant
#   ./start.sh --debug    also save every heard phrase to logs/phrases
cd "$(dirname "$0")" || exit 1
CHECK='import sys, tkinter; sys.exit(sys.version_info < (3, 12))'

# The venv's own Python once it exists, otherwise Python 3.12+ with tkinter from this computer
# (Homebrew's python3.14 has no tkinter unless python-tk@3.14 is installed, so 3.13 comes first)
for py in .venv/bin/python python3.13 python3.12 python3.14 python3 \
          /opt/homebrew/bin/python3.13 /usr/local/bin/python3.13; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "$CHECK" >/dev/null 2>&1; then
        exec "$py" bootstrap.py "$@"
    fi
done

echo "Python 3.12 or newer with tkinter wasn't found."
if [ "$(uname)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    printf "Install Python 3.13 with Tk now (brew install python@3.13 python-tk@3.13)? [Y/n] "
    read -r answer
    case "$answer" in
        [nN]*) ;;
        *) brew install python@3.13 python-tk@3.13 && exec "$(brew --prefix)/bin/python3.13" bootstrap.py "$@" ;;
    esac
    echo "Install it with: brew install python@3.13 python-tk@3.13"
elif [ "$(uname)" = "Darwin" ]; then
    echo "Install it from https://www.python.org/downloads/ (or Homebrew: brew install python@3.13 python-tk@3.13)."
else
    echo "Install it with your package manager, e.g.: sudo apt install python3 python3-venv python3-tk libportaudio2"
fi
exit 1
