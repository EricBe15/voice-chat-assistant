@echo off
rem Sets everything up without starting the assistant: .venv, packages, models, llama.cpp.
rem start.bat does the same on its first run, so this is optional.
call "%~dp0start.bat" --install-only %*
