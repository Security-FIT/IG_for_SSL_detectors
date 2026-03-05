# What do deepfake speech detectors actually learn?

This repository contains the source code for the Interspeech 2026 submission: *"What do deepfake speech detectors actually learn?"*

It provides a framework for training, evaluating, and interpreting speech deepfake detectors using Integrated Gradients (IG) to understand what features (and artefacts) self-supervised learning (SSL) models rely on to make their decisions.

## Project Structure

*   **models/**: Contains the model architectures (`aasist.py`, `mhfa.py`, `sls.py`, `wavlm_aasist.py`, `wavlm_camhfa.py`, `wavlm_sls.py`).
*   **utils/**: Utility scripts for data loading (`asvspoof5_dataset.py`), processing (`audio_utils.py`), metrics (`metrics.py`), and visualization/Integrated Gradients support (`ig_utils.py`, `ig_visualization.py`).
*   **augmentation/**: Code for augmenting speech data (`RawBoost`, `Codec`, `NoiseFilter`, etc.).
*   **artefacts_check/**: Tools to compute and analyze correlations and Equal Error Rates (EER) of specific audio artefacts (`compute_artefact_correlations.py`, `compute_artefact_stats.py`).
*   **labelling_app/**: A PHP-based web application for manually annotating/labeling audio features and artefacts.
*   **scores/**: Scripts for error rate calculation and model fusion scoring.
*   **Root Scripts**:
    *   `train.py` & `eval.py`: Main scripts for training and evaluating the deepfake detectors.
    *   `compute_ssl_means.py`: Script to compute mean SSL representations.
    *   `generate_ig_plots.py` & `plot_combined_ig.py`: Scripts for generating Integrated Gradients visualizations.
    *   `download_model.py`: Script to optionally pull base models/weights.

## Usage

1.  **Environment Setup**:
    Initialize the required dependencies using the provided environment scripts:
    ```bash
    # Linux/macOS
    source env/setup_env.sh
    # Windows
    env\setup_env.bat
    ```
    *(Alternatively, install packages directly from `env/requirements.txt`)*

2.  **Training**:
    Configure your paths in `config.py` and run:
    ```bash
    python train.py --model camhfa
    ```
    
    **Arguments for `train.py`:**
    *   `--model`: Model architecture to train (`sls`, `camhfa`, or `aasist`). **[Required]**
    *   `--data_dir`: Path to ASVspoof5 root directory (defaults to `config.DATA_DIR`).
    *   `--train_protocol`: Train protocol filename (defaults to `config.TRAIN_PROTOCOL`).
    *   `--dev_protocol`: Dev protocol filename (defaults to `config.DEV_PROTOCOL`).
    *   `--output_dir`: Directory to save checkpoints (defaults to `config.OUTPUT_DIR`).
    *   `--epochs`: Number of training epochs (default: `10`).
    *   `--batch_size`: Batch size per GPU (default: `4`).
    *   `--device`: Computation device (`cuda` or `cpu`).
    *   `--augment`: Flag to apply data augmentation.
    *   `--freeze_wavlm`: Flag to freeze the WavLM backbone entirely.
    *   `--warmup_epochs`: Number of epochs to freeze WavLM before switching to end-to-end training (default: `0`).

3.  **Generating Integrated Gradients (IG) Explanations**:
    Visualize the learned representations using IG plots:
    ```bash
    python generate_ig_plots.py --input_csv selections.csv --audio_dir /path/to/flac/
    ```
    
    **Arguments for `generate_ig_plots.py`:**
    *   `--input_csv`: Path to a CSV or text list containing `FileID` strings to process. **[Required]**
    *   `--audio_dir`: Root directory containing the audio `.flac` files. **[Required]**
    *   `--output_dir`: Directory to save the output `.png` plots and interactive `.json` data (default: `outputs/plots`).
    *   `--models_dir`: Directory containing the `.pt` model checkpoints (default: `models`).

## Paper Context
This repository provides the experimental framework addressing the core question of our IS26 submission: establishing exactly what cues state-of-the-art self-supervised and end-to-end deepfake systems utilize when distinguishing bona fide speech from spoofed audio.
