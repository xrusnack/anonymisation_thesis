# Deidentification Tool

This tool is implemented as practical part of a thesis: https://is.muni.cz/th/rf3hj/Anonymisation_of_Clinical_Notes.pdf

## Dependencies:
The script requires Python with version 3.8.*.
If you do not have this version available on your system, please install it (e.g., from source or using conda).

To resolve and install all dependencies required to run deidentification_tool.py, navigate to the directory containing the script.
If you do not have Poetry installed, first run:

```bash
pip install poetry
```

Then resolve the dependencies by running:

```bash
poetry install
```

## Usage
To use the tool, open the deidentification_tool.py file and fill in the neccesary fields as instructed. 
Then run:

```bash
poetry run python deidentification_tool.py
```
