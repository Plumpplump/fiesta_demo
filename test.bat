:: Author: David Li, SISHK
@echo off
setlocal
set "VENV=C:\Users\david\OneDrive\Python Coding\myvenv"
cd /d "%~dp0"
"%VENV%\Scripts\python.exe" -m pytest -q %*