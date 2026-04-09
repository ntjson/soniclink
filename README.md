# SonicLink

Install the project into the repo virtual environment with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

Run the behavioral test suite with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Generate the Phase 2 transmitter WAV with:

```powershell
.\.venv\Scripts\python.exe -m acoustic_modem.cli tx --text HELLO --out hello.wav
```
