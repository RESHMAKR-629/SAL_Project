I apologize for the confusion. Here is the **complete, raw Markdown code** for the entire `README.md` file.

You can copy the code block below **as a single text** and paste it directly into a file named `README.md`.

````markdown
# Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech

This repository contains the implementation and analysis code for the paper:

**"Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech"**
*Karl El Hajal, Enno Hermann, Sevada Hovsepyan, Mathew Magimai.-Doss (Idiap Research Institute)*
[arXiv:2506.01618v1](https://arxiv.org/abs/2506.01618v1)

This project explores using unsupervised rhythm and voice conversion to transform dysarthric speech into healthy-like speech to improve Automatic Speech Recognition (ASR) performance. It includes pipelines for speech conversion, ASR fine-tuning (Whisper & Wav2Vec2), and detailed linguistic analysis.

---

## 📋 Table of Contents
1. [Installation & Setup](#1-installation--setup)
2. [Data Preparation](#2-data-preparation)
3. [Speech Conversion Pipeline](#3-speech-conversion-pipeline)
4. [ASR Training (LOSO)](#4-asr-training-loso)
5. [Analysis & Evaluation](#5-analysis--evaluation)
6. [Kaldi Baseline (LF-MMI)](#6-kaldi-baseline-lf-mmi)
7. [Citation](#citation)

---

## 1. Installation & Setup

This project builds upon the **RnV (Rhythm and Voice)** framework.

### Step 1: Clone the RnV Repository
First, clone the base repository and follow their installation instructions to set up the environment and download necessary checkpoints (WavLM, HiFi-GAN vocoder, etc.).

```bash
git clone [https://github.com/idiap/RnV.git](https://github.com/idiap/RnV.git)
cd RnV
# Follow the instructions in the RnV README to install dependencies
# e.g., pip install -r requirements.txt
# Download checkpoints to a 'checkpoints/' directory
````

### Step 2: Install Additional Dependencies

For the ASR training and linguistic analysis scripts in this repo, you will need additional libraries:

```bash
pip install transformers datasets torch librosa soundfile jiwer pandas matplotlib seaborn g2p_en nltk tqdm scipy tabulate
```

-----

## 2\. Data Preparation

### TORGO Dataset

1.  Download the [TORGO database](https://www.google.com/search?q=http://www.cs.toronto.edu/~torgo/).
2.  Organize the audio files into a structure separating Dysarthric (`F_Dys`, `M_Dys`) and Control (`F_Con`, `M_Con`) speakers.
3.  Ensure you have the text transcripts available.

**Recommended Structure:**

```text
data/
├── torgo/
│   ├── F_Dys/
│   ├── M_Dys/
│   ├── F_Con/
│   └── M_Con/
└── transcripts/
    ├── F01.txt
    ├── M01.txt
    ...
```

-----

## 3\. Speech Conversion Pipeline

We provide scripts to convert dysarthric speech into "healthy" speech using various configurations (Rhythm conversion, Voice conversion, or both).

### Single File Conversion

To test the conversion on a single audio file, use `conversion.py`.
*Note: Open `conversion.py` and update the `CHECKPOINTS_DIR`, `source_wav_path`, and `target_style_feats_path` variables before running.*

```bash
python conversion.py
```

### Batch Conversion (All Experiments)

To generate the full dataset for all experimental conditions (Original, Vocoded, kNN-VC, Urhythmic, Syllable, etc.), use `conversion_all.py`.

**Configuration:**
Open `conversion_all.py` and update the constants at the top of the file to match your local paths:

  * `CHECKPOINTS_DIR`: Path to RnV checkpoints.
  * `MAIN_PATH`: Path to your raw TORGO audio.
  * `OUTPUT_BASE`: Where to save the converted datasets.

**Run:**

```bash
python conversion_all.py
```

This will generate subfolders for each experiment (e.g., `experiments/Syllable_Global_kNN-VC/F01/...`) containing the processed audio.

-----

## 4\. ASR Training (LOSO)

We perform **Leave-One-Speaker-Out (LOSO)** cross-validation training. For each dysarthric speaker, the model is trained on all other speakers and tested on that specific speaker.

### Whisper Fine-Tuning

Train OpenAI's Whisper (Base) model using `train_whisper.py`.

```bash
python train_whisper.py \
  --dataset_dir "/path/to/converted_data/Experiment_Name" \
  --transcript_dir "/path/to/transcripts" \
  --output_dir "./results_whisper" \
  --experiment_name "Syllable_Global_kNN-VC" \
  --gpu_id 0
```

### Wav2Vec2 Fine-Tuning

Train Facebook's Wav2Vec2 (Base) model using `train_wav.py`.
*Note: This script includes stability fixes (FP32, padding, filtering) specifically designed to handle dysarthric speech without crashing.*

```bash
python train_wav.py \
  --dataset_dir "/path/to/converted_data/Experiment_Name" \
  --transcript_dir "/path/to/transcripts" \
  --output_dir "./results_wav2vec" \
  --experiment_name "Syllable_Global_kNN-VC" \
  --gpu_id 0
```

**Output:** Both scripts create a `predictions/` folder inside the experiment directory containing CSVs with reference vs. predicted text for analysis.

-----

## 5\. Analysis & Evaluation

We provide a comprehensive analysis suite to evaluate Word Error Rate (WER), Phoneme Error Rate (PER), Syllable Deviation, and Error Composition (Insertion/Deletion/Substitution).

### Master Analysis Script

This script scans all your experiment folders (Whisper and Wav2Vec2), generates a master results table, and produces comparative plots (Heatmaps, Bar Charts).

```bash
python analyze_and_plot_master.py --experiments_root "./results_whisper"
```

**What this generates:**

1.  `MASTER_ALL_EXPERIMENTS.csv`: A summary table of all metrics for all speakers.
2.  `aggregated_plots/`: Visualizations including:
      * **WER Heatmaps**: Performance per speaker vs. experiment.
      * **Error Composition**: Stacked bars showing insertions vs. deletions.
      * **Linguistic Analysis**: Correlation between speaking rate and WER.

### Additional Linguistic Insights

For deeper analysis (Hallucinations, Severity Impact, POS Tagging):

```bash
python generate_linguistic_insights.py \
  --experiments_root "./experiments" \
  --master_csv "./experiments/MASTER_ALL_EXPERIMENTS.csv" \
  --stats_csv "speaker_stats.csv"
```

-----

## 6\. Kaldi Baseline (LF-MMI)

The `run_loso.sh` script implements the Lattice-Free MMI (LF-MMI) baseline using the Kaldi toolkit, as described in the paper.

**Prerequisites:**

  * Kaldi installed and compiled.
  * `path.sh` and `cmd.sh` configured for your system.

**Procedure:**

1.  **Data Preparation:** The script generates `wav.scp`, `text`, `utt2spk` for the LOSO splits.
2.  **Feature Extraction:** Extracts MFCCs and computes CMVN.
3.  **GMM-HMM Training:** Trains Monophone, Triphone (delta), and LDA-MLLT models to generate alignments.
4.  **Chain Training (LF-MMI):**
      * Generates numerator and denominator lattices.
      * Trains the TDNN-F acoustic model.
5.  **Decoding:** Decodes using a bigram language model (for sentences) or grammar (for isolated words).

**Usage:**

```bash
./run_loso.sh --stage 0 --data_dir "/path/to/torgo"
```

-----

## Citation

If you use this code or the analysis methods, please cite the original paper:

```bibtex
@article{elhajal2025unsupervised,
  title={Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech},
  author={El Hajal, Karl and Hermann, Enno and Hovsepyan, Sevada and Magimai-Doss, Mathew},
  journal={arXiv preprint arXiv:2506.01618},
  year={2025}
}
```

```
```
