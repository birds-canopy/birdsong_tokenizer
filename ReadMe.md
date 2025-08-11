# Tokenizing Canary Vocal Chunks

This repository contains the code and resources used to process and analyze canary vocal recordings.  
It is part of a scientific poster focusing on the extraction, tokenization, and analysis of vocal sequences.

## Project Structure

- `src/` : Python scripts defining the functions used in the notebook (audio preprocessing, segmentation, tokenization, etc.).
- `tokenizing_canary_vocal_chunks.ipynb` : Main notebook illustrating the full analysis workflow.
- `requirements.txt` : Minimal list of required dependencies.
- (other folders to specify: `data/`, `results/`, etc.)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/<username>/<repo-name>.git
    cd <repo-name>
    ```

2. Create a virtual environment and install the dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows
    pip install -r requirements.txt
    ```

## The notebook follows these main steps:

1. Load and preprocess annotation files.

2. Tokenize songs phrases into vocal “chunks”.

3. Analyze and visualize the results.

## Using the Python scripts

The scripts in the src/ folder can be imported to run specific parts of the pipeline:

from src.my_module import my_function

## References

If this repository is linked to a scientific article, please cite:

...

## License

This project is distributed under the MIT license.