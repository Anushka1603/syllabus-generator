# AI Syllabus Generator

A Python command-line tool that generates a complete academic syllabus using the `qwen3:4b` model through Ollama. The output is structured around Bloom's Taxonomy.

## What It Generates
- Course objectives
- Course outcomes with Bloom's tags
- Program objectives
- Program outcomes with Bloom's tags
- Unit-wise syllabus
- Textbook and reference book suggestions
- YouTube learning suggestions
- Optional text file export

## Requirements
- Python 3.10+
- Ollama installed locally
- `llama3.1:8b` pulled in Ollama

## Setup

### 1. Create a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Python packages
```powershell
pip install -r requirements.txt
```

### 3. Start Ollama
```powershell
ollama serve
```

### 4. Pull the model
```powershell
ollama pull llama3.1:8b
```

## Run
```powershell
python syllabus_generator.py
```

The script will ask for:
- academic level
- program name
- course name
- number of units
- whether to save the generated result to a file

## Notes
- Ollama must be running at `http://localhost:11434`
- The script saves output as a UTF-8 text file when you choose the save option
