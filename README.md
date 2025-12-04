# SAL_Project

# Unsupervised Rhythm and Voice Conversion for Dysarthric ASR

This repository contains the implementation and analysis code for the paper:

**"Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech"** *Karl El Hajal, Enno Hermann, Sevada Hovsepyan, Mathew Magimai.-Doss (Idiap Research Institute)* [arXiv:2506.01618v1](https://arxiv.org/abs/2506.01618v1)

This project explores using unsupervised rhythm and voice conversion to transform dysarthric speech into healthy-like speech to improve Automatic Speech Recognition (ASR) performance. It includes pipelines for speech conversion, ASR fine-tuning (Whisper & Wav2Vec2), and detailed linguistic analysis.

---

## 📋 Table of Contents
1. [Installation & Setup](#1-installation--setup)
2. [Data Preparation](#2-data-preparation)
3. [Speech Conversion Pipeline](#3-speech-conversion-pipeline)
4. [ASR Training (LOSO)](#4-asr-training-loso)
5. [Analysis & Evaluation](#5-analysis--evaluation)
6. [Kaldi Baseline (LF-MMI)](#6-kaldi-baseline-lf-mmi)

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

### **Step 2: Install Additional Dependencies**

For the ASR training and linguistic analysis scripts in this repo, you will need additional libraries:

pip install transformers datasets torch librosa soundfile jiwer pandas matplotlib seaborn g2p_en nltk tqdm

### **3. Data Preparation**

```markdown
## 📂 Data Preparation

1.  **Download Data:** Obtain the [TORGO database](http://www.cs.toronto.edu/~torgo/). [cite: 11, 101]
2.  **Organize Files:** Structure your audio files to separate Dysarthric and Control speakers.
3.  **Transcripts:** Ensure you have text transcripts for all audio files.

**Recommended Directory Structure:**
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

### **4. Speech Conversion**

```markdown
## 🔄 Speech Conversion Pipeline

We provide scripts to convert dysarthric speech into "healthy" speech using various configurations (Rhythm conversion, Voice conversion, or both). [cite: 42, 43, 53]

### Single File Conversion (`conversion.py`)
To test the conversion on a single audio file:
1.  Open `conversion.py`.
2.  Update the `CHECKPOINTS_DIR`, `source_wav_path`, and `target_style_feats_path` variables.
3.  Run the script:

```bash
python conversion.py

Batch Conversion (conversion_all.py)

To generate the full dataset for all experimental conditions (Original, Vocoded, kNN-VC, Urhythmic, Syllable, etc.):

    Open conversion_all.py.

    Update the constants at the top (MAIN_PATH, OUTPUT_BASE, etc.) to match your local paths.

    Run the script:
        python conversion_all.py
Here is the complete README.md content for your GitHub repository, structured section by section. You can copy each block directly.
1. Header & Introduction
Markdown

# Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech

This repository contains the implementation and analysis code for the paper:

**"Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech"** *Karl El Hajal, Enno Hermann, Sevada Hovsepyan, Mathew Magimai.-Doss (Idiap Research Institute)* [arXiv:2506.01618v1](https://arxiv.org/abs/2506.01618v1)

This project explores using unsupervised rhythm and voice conversion to transform dysarthric speech into healthy-like speech to improve Automatic Speech Recognition (ASR) performance. [cite_start]It includes pipelines for speech conversion, ASR fine-tuning (Whisper & Wav2Vec2), and detailed linguistic analysis. [cite: 1, 2, 6, 8, 9]

---

2. Installation
Markdown

## 🛠️ Installation & Setup

This project builds upon the **RnV (Rhythm and Voice)** framework.

### 1. Clone the Base Repository
First, clone the original RnV repository and follow their instructions to set up the environment and download necessary checkpoints (WavLM, HiFi-GAN vocoder, etc.).

```bash
git clone [https://github.com/idiap/RnV.git](https://github.com/idiap/RnV.git)
cd RnV
# Follow the RnV README to install dependencies (e.g., pip install -r requirements.txt)
# Download checkpoints to a 'checkpoints/' directory

2. Install Project Dependencies

Install the additional libraries required for ASR training and analysis:
Bash

pip install transformers datasets torch librosa soundfile jiwer pandas matplotlib seaborn g2p_en nltk tqdm


### **3. Data Preparation**

```markdown
## 📂 Data Preparation

1.  **Download Data:** Obtain the [TORGO database](http://www.cs.toronto.edu/~torgo/). [cite: 11, 101]
2.  **Organize Files:** Structure your audio files to separate Dysarthric and Control speakers.
3.  **Transcripts:** Ensure you have text transcripts for all audio files.

**Recommended Directory Structure:**
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


### **4. Speech Conversion**

```markdown
## 🔄 Speech Conversion Pipeline

We provide scripts to convert dysarthric speech into "healthy" speech using various configurations (Rhythm conversion, Voice conversion, or both). [cite: 42, 43, 53]

### Single File Conversion (`conversion.py`)
To test the conversion on a single audio file:
1.  Open `conversion.py`.
2.  Update the `CHECKPOINTS_DIR`, `source_wav_path`, and `target_style_feats_path` variables.
3.  Run the script:

```bash
python conversion.py

Batch Conversion (conversion_all.py)

To generate the full dataset for all experimental conditions (Original, Vocoded, kNN-VC, Urhythmic, Syllable, etc.):

    Open conversion_all.py.

    Update the constants at the top (MAIN_PATH, OUTPUT_BASE, etc.) to match your local paths.

    Run the script:

Bash

python conversion_all.py

This will create processed datasets for every experiment type (e.g., Syllable_Global_kNN-VC) in your output directory.

### **5. ASR Training**

```markdown
## 🚀 ASR Training (LOSO)

We perform **Leave-One-Speaker-Out (LOSO)** cross-validation. The scripts automatically train on all available speakers *except* the test speaker, then evaluate on that speaker. [cite: 142]

### Whisper Fine-Tuning
To fine-tune OpenAI's Whisper (Base) model: [cite: 149]

```bash
python train_whisper.py \
  --dataset_dir "/path/to/converted_data/Experiment_Name" \
  --transcript_dir "/path/to/transcripts" \
  --output_dir "./results_whisper" \
  --experiment_name "Whisper_Syllable_Global" \
  --gpu_id 0

### Wav2Vec2 Fine-Tuning

To fine-tune Facebook's Wav2Vec2 (Base) model. This script includes stability fixes (FP32, padding) for dysarthric speech:
 ```bash
python train_wav.py \
  --dataset_dir "/path/to/converted_data/Experiment_Name" \
  --transcript_dir "/path/to/transcripts" \
  --output_dir "./results_wav2vec" \
  --experiment_name "Wav2Vec_Syllable_Global" \
  --gpu_id 0


### **6. Kaldi Baseline**

```markdown
## 📊 Kaldi Baseline (LF-MMI)

The `run_loso.sh` script implements the Lattice-Free MMI (LF-MMI) baseline using the Kaldi toolkit. [cite: 145, 146]

**Prerequisites:**
* Kaldi installed and compiled.
* `path.sh` and `cmd.sh` configured.

**Usage:**
```bash
./run_loso.sh --stage 0 --data_dir "/path/to/torgo"

Steps Performed:

    Data Prep: Generates wav.scp, text, utt2spk.

    Feature Extraction: MFCCs + CMVN.

    GMM-HMM: Monophone & Triphone training for alignments.

    Chain Training: TDNN-F acoustic model training using LF-MMI objective.

    Decoding: Evaluation using bigram language models (sentences) or grammar (isolated words).

### **7. Citation**

```markdown
## 📄 Citation

If you use this code, please cite the original paper:

```bibtex
@article{elhajal2025unsupervised,
  title={Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech},
  author={El Hajal, Karl and Hermann, Enno and Hovsepyan, Sevada and Magimai-Doss, Mathew},
  journal={arXiv preprint arXiv:2506.01618},
  year={2025}
}
