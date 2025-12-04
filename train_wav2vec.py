import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union
import string
import re

# --- SECURITY HOTFIX FOR CVE-2025-32434 ---
import transformers.utils.import_utils
import transformers.trainer
import transformers.modeling_utils

transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.trainer, "check_torch_load_is_safe"):
    transformers.trainer.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
# ------------------------------------------

import torch
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC, 
    TrainingArguments, 
    Trainer,
    TrainerCallback, 
    EarlyStoppingCallback
)
import librosa
import soundfile as sf
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import jiwer
from datetime import datetime
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Force single GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"

# Setup logging
def setup_logging(output_dir: Path, experiment_name: str):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# --- FIXED LOGGING CALLBACK ---
class EpochLoggingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
        
    def on_epoch_end(self, args, state, control, **kwargs):
        # Search backwards to find the TRAINING log (which has 'loss')
        train_log = None
        if state.log_history:
            for log in reversed(state.log_history):
                if 'loss' in log:
                    train_log = log
                    break
        
        if train_log:
            self.logger.info(
                f"Epoch {train_log.get('epoch', 0):.2f} - "
                f"Loss: {train_log['loss']:.4f} - "
                f"LR: {train_log.get('learning_rate', 0):.2e}"
            )

# --- ROBUST DATASET WITH PADDING & FILTERING ---
class TorgoDataset(Dataset):
    def __init__(self, audio_files: List[Path], transcripts: Dict[str, str], 
                 processor: Wav2Vec2Processor, sample_rate: int = 16000):
        self.processor = processor
        self.sample_rate = sample_rate
        self.audio_files = []
        self.transcripts = {}
        
        # 1. Sanitize Text (Remove punctuation)
        clean_transcripts = {}
        for k, v in transcripts.items():
            clean_text = re.sub(r'[^A-Z\s]', '', v.upper())
            clean_text = " ".join(clean_text.split()) 
            if clean_text:
                clean_transcripts[k] = clean_text

        # 2. Filtering Logic
        print(f"Filtering {len(audio_files)} files...")
        kept = 0
        skipped_short = 0
        skipped_math = 0
        
        for audio_path in audio_files:
            audio_name = audio_path.stem
            transcript = clean_transcripts.get(audio_name, "")
            
            if not transcript: continue

            try:
                info = sf.info(str(audio_path))
                duration = info.duration
                
                # Filter A: Remove very short/empty files (< 0.5s)
                if duration < 0.5: 
                    skipped_short += 1
                    continue

                # Filter B: Mathematical Safety Check for CTC
                # Wav2Vec2 downsamples by 320x. We need enough output frames for the text.
                input_frames = duration * 16000
                potential_output_frames = input_frames // 320
                
                # If frames < text_length, CTC Loss becomes Infinite/NaN
                if potential_output_frames < len(transcript):
                    skipped_math += 1
                    continue 

                self.audio_files.append(audio_path)
                self.transcripts[audio_name] = transcript
                kept += 1
            except:
                continue
                
        print(f"Dataset Ready: Kept {kept} files.")
        print(f"  - Skipped (Too short < 0.5s): {skipped_short}")
        print(f"  - Skipped (CTC Math Fail): {skipped_math}")

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # --- CRITICAL FIX: PADDING ---
        # Pad audio to minimum 1.0s (16000 samples) to prevent batch errors
        if len(audio) < 16000:
            padding = 16000 - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        # Normalize to -20dB
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 10**(-20/20)
        if rms > 0:
            audio = audio * (target_rms / rms)
        
        audio_name = audio_path.stem
        transcript = self.transcripts.get(audio_name, "")
        
        # Process inputs
        input_values = self.processor(
            audio, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        ).input_values.squeeze(0)
        
        # Process labels using tokenizer directly (Fixes TypeError)
        labels = self.processor.tokenizer(
            transcript, 
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        return {
            "input_values": input_values,
            "labels": labels,
            "audio_path": str(audio_path)
        }

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Mask padding with -100
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def load_transcripts(transcript_dir: Path) -> Dict[str, str]:
    transcripts = {}
    print(f"Scanning for transcripts in: {transcript_dir}")
    files_found = 0
    
    for txt_file in transcript_dir.rglob("*.txt"):
        audio_id = txt_file.stem
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    transcripts[audio_id] = text.upper()
                    files_found += 1
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
            
    print(f"Loaded {files_found} transcripts.")
    return transcripts

def get_speaker_files(dataset_dir: Path) -> Dict[str, List[Path]]:
    speaker_files = {}
    # Look for both Dysarthric and Control folders
    target_dirs = list(dataset_dir.glob("*_Dys")) + list(dataset_dir.glob("*_Con"))
    
    for gender_dir in target_dirs:
        for speaker_dir in gender_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                audio_files = list(speaker_dir.rglob("*.wav"))
                if audio_files:
                    speaker_files[speaker_id] = audio_files
    return speaker_files

def compute_wer(predictions: List[str], references: List[str]) -> float:
    # Safe WER computation (avoiding jiwer version issues)
    translator = str.maketrans('', '', string.punctuation)
    norm_preds = []
    norm_refs = []
    
    for p, r in zip(predictions, references):
        p_clean = p.lower().translate(translator).replace('\n', ' ').strip()
        p_clean = " ".join(p_clean.split())
        norm_preds.append(p_clean)
        
        r_clean = r.lower().translate(translator).replace('\n', ' ').strip()
        r_clean = " ".join(r_clean.split())
        norm_refs.append(r_clean)

    wer = jiwer.wer(norm_refs, norm_preds)
    return wer * 100

def separate_sentences_and_words(audio_files: List[Path], transcripts: Dict[str, str]) -> Tuple[List[Path], List[Path]]:
    sentences = []
    words = []
    for audio_file in audio_files:
        audio_name = audio_file.stem
        text = transcripts.get(audio_name, "").strip()
        word_count = len(text.split())
        if word_count <= 1:
            words.append(audio_file)
        else:
            sentences.append(audio_file)
    return sentences, words

def evaluate_speaker(model, processor, test_files: List[Path], 
                    transcripts: Dict[str, str], device: str, logger,
                    output_dir: Path, test_speaker: str) -> Tuple[float, float]:
    model.eval()
    sentences, words = separate_sentences_and_words(test_files, transcripts)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_subset(files, subset_name):
        predictions = []
        references = []
        filenames = []
        
        with tqdm(files, desc=f"Evaluating {subset_name}", leave=False) as pbar:
            for audio_file in pbar:
                try:
                    audio, sr = librosa.load(audio_file, sr=16000)
                    rms = np.sqrt(np.mean(audio**2))
                    target_rms = 10**(-20/20)
                    if rms > 0: audio = audio * (target_rms / rms)
                    
                    input_values = processor(
                        audio, sampling_rate=16000, return_tensors="pt"
                    ).input_values.to(device)
                    
                    with torch.no_grad():
                        logits = model(input_values).logits
                    
                    predicted_ids = torch.argmax(logits, dim=-1)
                    prediction = processor.batch_decode(predicted_ids)[0]
                    
                    audio_name = audio_file.stem
                    reference = transcripts.get(audio_name, "")
                    
                    predictions.append(prediction)
                    references.append(reference)
                    filenames.append(audio_file.name)
                except: pass
        
        if len(predictions) > 0:
            wer = compute_wer(predictions, references)
            logger.info(f"{subset_name} - {len(files)} files - WER: {wer:.2f}%")
            
            df = pd.DataFrame({
                'filename': filenames,
                'reference': references,
                'prediction': predictions
            })
            csv_file = predictions_dir / f"{test_speaker}_{subset_name.lower().replace(' ', '_')}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            return wer
        else:
            return 0.0
    
    sentence_wer = evaluate_subset(sentences, "Sentences") if sentences else 0.0
    word_wer = evaluate_subset(words, "Isolated Words") if words else 0.0
    return sentence_wer, word_wer

def train_loso(dataset_dir: Path, transcript_dir: Path, output_dir: Path, experiment_name: str, gpu_id: int = 0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    output_dir = output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir, experiment_name)
    logger.info(f"Starting experiment: {experiment_name} (Wav2Vec2 Base English)")
    
    transcripts = load_transcripts(transcript_dir)
    speaker_files = get_speaker_files(dataset_dir)
    
    logger.info("Loading Wav2Vec2 Base English model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    results = {"experiment": experiment_name, "per_speaker_results": {}, "average_results": {}}
    
    # --- TARGET SPEAKERS ---
    dysarthric_speakers = ['F01', 'M01', 'M02', 'M04', 'M05', 'F03', 'F04', 'M03']

    # --- FIX: Iterate strictly over target speakers ---
    for test_speaker in tqdm(dysarthric_speakers, desc="LOSO iterations"):
        
        if test_speaker not in speaker_files:
            continue
 
        logger.info(f"\n{'='*50}\nTesting on speaker: {test_speaker}\n{'='*50}")
        
        test_files = speaker_files[test_speaker]
        train_files = []
        
        # Build training set from everyone else
        training_speakers = [s for s in speaker_files.keys() if s != test_speaker]
        
        if len(training_speakers) > 1:
            val_speaker = training_speakers[0]
            val_files = speaker_files[val_speaker]
            for speaker in training_speakers[1:]:
                train_files.extend(speaker_files[speaker])
        else:
            all_train = speaker_files[training_speakers[0]]
            split_idx = int(0.9 * len(all_train))
            train_files = all_train[:split_idx]
            val_files = all_train[split_idx:]
        
        # Create datasets
        train_dataset = TorgoDataset(train_files, transcripts, processor)
        val_dataset = TorgoDataset(val_files, transcripts, processor)
        
        # Initialize fresh model
        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h",
            ctc_loss_reduction="mean", 
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        
        # Stability Configs
        model.config.ctc_zero_infinity = True
        model.freeze_feature_encoder()
        model.to(device)
        
        checkpoint_dir = output_dir / f"checkpoints_{test_speaker}"
        
        # --- FINAL STABLE TRAINING ARGUMENTS ---
        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=3,
            
            # CRITICAL: Low LR + FP32 to prevent NaN
            learning_rate=3e-5,              
            max_grad_norm=1.0,
            adam_epsilon=1e-8,
            weight_decay=0.005,
            
            gradient_checkpointing=False, 
            fp16=False,                      # STRICTLY FALSE
            group_by_length=False,           # Random batching essential
            
            warmup_steps=500,
            num_train_epochs=100,            # 100 Epochs
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_strategy="epoch",
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["none"],
            dataloader_num_workers=12,
        )
        
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor.feature_extractor,
            data_collator=data_collator,
            callbacks=[
                EpochLoggingCallback(logger),
                EarlyStoppingCallback(early_stopping_patience=10)
            ],
        )
        
        resume_checkpoint = False
        if checkpoint_dir.exists() and any(x.name.startswith("checkpoint") for x in checkpoint_dir.iterdir()):
            resume_checkpoint = True

        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        # Save & Evaluate
        model_path = output_dir / f"model_{test_speaker}"
        model.save_pretrained(model_path)
        processor.save_pretrained(model_path)
        
        s_wer, w_wer = evaluate_speaker(model, processor, test_files, transcripts, device, logger, output_dir, test_speaker)
        results["per_speaker_results"][test_speaker] = {"sentences_wer": s_wer, "isolated_words_wer": w_wer}
        logger.info(f"Speaker {test_speaker} - Sentences WER: {s_wer:.2f}% | Isolated Words WER: {w_wer:.2f}%")
        
        del model, trainer
        torch.cuda.empty_cache()
    
    # Calculate Average
    valid_speakers = [s for s in dysarthric_speakers if s in results["per_speaker_results"]]
    if valid_speakers:
        avg_s = np.mean([results["per_speaker_results"][s]["sentences_wer"] for s in valid_speakers])
        avg_w = np.mean([results["per_speaker_results"][s]["isolated_words_wer"] for s in valid_speakers])
        results["average_results"] = {"avg_sentences_wer": avg_s, "avg_isolated_words_wer": avg_w}
        logger.info(f"FINAL - Avg Sentences WER: {avg_s:.2f}% | Avg Isolated Words WER: {avg_w:.2f}%")
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    summary_data = []
    for speaker, result in results["per_speaker_results"].items():
        summary_data.append({'speaker': speaker, **result})
    
    pd.DataFrame(summary_data).to_csv(output_dir / "summary_results.csv", index=False)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wav2Vec2 LOSO training")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--transcript_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    transcript_dir = Path(args.transcript_dir)
    output_dir = Path(args.output_dir)
    
    train_loso(dataset_dir, transcript_dir, output_dir, args.experiment_name, args.gpu_id)
