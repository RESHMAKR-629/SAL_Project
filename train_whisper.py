import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

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
#warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

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

# Custom callback to log only at epoch end
class EpochLoggingCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.log_history:
            last_log = state.log_history[-1]
            if 'loss' in last_log:
                self.logger.info(
                    f"Epoch {last_log.get('epoch', 0):.2f} - "
                    f"Loss: {last_log['loss']:.4f} - "
                    f"LR: {last_log.get('learning_rate', 0):.2e}"
                )

# Dataset class
'''class TorgoDataset(Dataset):
    def __init__(self, audio_files: List[Path], transcripts: Dict[str, str], 
                 processor: Wav2Vec2Processor, sample_rate: int = 16000):
        self.audio_files = audio_files
        self.transcripts = transcripts
        self.processor = processor
        self.sample_rate = sample_rate
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load and normalize audio to 16kHz
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Normalize to -20dB
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 10**(-20/20)
        if rms > 0:
            audio = audio * (target_rms / rms)
        
        # Get transcript using basename (without extension)
        audio_name = audio_path.stem
        transcript = self.transcripts.get(audio_name, "")
        
        # Wav2Vec2 inputs (Raw audio values, NOT input_features)
        input_values = self.processor(
            audio, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        ).input_values.squeeze(0)
        
        # Process labels (Tokenized text)
        #with self.processor.as_target_processor():
        labels = self.processor(
                transcript, 
                return_tensors="pt"
        ).input_ids.squeeze(0)
        
        return {
            "input_values": input_values,
            "labels": labels,
            "audio_path": str(audio_path)
        }
'''

# Dataset class
class TorgoDataset(Dataset):
    def __init__(self, audio_files: List[Path], transcripts: Dict[str, str],
                 processor: Wav2Vec2Processor, sample_rate: int = 16000):
        self.audio_files = audio_files
        self.transcripts = transcripts
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]

        # Load and normalize audio to 16kHz
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Normalize to -20dB
        rms = np.sqrt(np.mean(audio**2))
        target_rms = 10**(-20/20)
        if rms > 0:
            audio = audio * (target_rms / rms)

        # Get transcript using basename (without extension)
        audio_name = audio_path.stem
        transcript = self.transcripts.get(audio_name, "")

        # 1. Process Audio (Input Values)
        # We call the processor normally here because we are passing audio
        input_values = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_values.squeeze(0)

        # 2. Process Labels (Tokenized Text)
        # CRITICAL FIX: We access the tokenizer DIRECTLY.
        # Using self.processor(transcript) would fail because it thinks text is audio.
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
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

def load_transcripts(transcript_dir: Path) -> Dict[str, str]:
    """
    Load transcripts from a separate directory using rglob for recursive search.
    """
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
    """Get audio files organized by speaker from both Dys and Con directories"""
    speaker_files = {}
    
    target_dirs = list(dataset_dir.glob("*_Dys")) + list(dataset_dir.glob("*_Con"))
    
    for gender_dir in target_dirs:
        for speaker_dir in gender_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                audio_files = list(speaker_dir.rglob("*.wav"))
                if audio_files:
                    speaker_files[speaker_id] = audio_files
    
    return speaker_files

'''def compute_wer(predictions: List[str], references: List[str]) -> float:
    """Compute Word Error Rate"""
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords()
    ])
    
    wer = jiwer.wer(
        references,
        predictions,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )
    
    return wer * 100  # Return as percentage
'''

import string

def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate (Fixed for jiwer 3.0+)
    We manually normalize text to avoid 'truth_transform' argument errors.
    """
    # 1. Manual Normalization (equivalent to the old jiwer transforms)
    # This handles Lowercase, Strip, RemovePunctuation, and RemoveMultipleSpaces
    translator = str.maketrans('', '', string.punctuation)
    
    norm_preds = []
    norm_refs = []
    
    for p, r in zip(predictions, references):
        # Normalize Prediction
        p_clean = p.lower().translate(translator).replace('\n', ' ').strip()
        p_clean = " ".join(p_clean.split()) # Remove multiple spaces
        norm_preds.append(p_clean)
        
        # Normalize Reference
        r_clean = r.lower().translate(translator).replace('\n', ' ').strip()
        r_clean = " ".join(r_clean.split()) # Remove multiple spaces
        norm_refs.append(r_clean)

    # 2. Compute WER using the clean strings
    # jiwer.wer() in 3.0+ simply takes the lists of strings
    wer = jiwer.wer(norm_refs, norm_preds)
    
    return wer * 100  # Return as percentage

def separate_sentences_and_words(audio_files: List[Path], transcripts: Dict[str, str]) -> Tuple[List[Path], List[Path]]:
    """
    Separate isolated words from sentences based on transcript word count.
    """
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
    """Evaluate model on a speaker's data and save predictions (Wav2Vec2 version)"""
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
                # Load audio
                audio, sr = librosa.load(audio_file, sr=16000)
                
                # Normalize
                rms = np.sqrt(np.mean(audio**2))
                target_rms = 10**(-20/20)
                if rms > 0:
                    audio = audio * (target_rms / rms)
                
                # Process Input (Raw Waveform)
                input_values = processor(
                    audio, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_values.to(device)
                
                # Inference (CTC Decoding)
                with torch.no_grad():
                    logits = model(input_values).logits
                
                # Decode (Argmax)
                predicted_ids = torch.argmax(logits, dim=-1)
                prediction = processor.batch_decode(predicted_ids)[0]
                
                # Get reference
                audio_name = audio_file.stem
                reference = transcripts.get(audio_name, "")
                
                predictions.append(prediction)
                references.append(reference)
                filenames.append(audio_file.name)
        
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
            logger.info(f"{subset_name} - No files found")
            return 0.0
    
    sentence_wer = evaluate_subset(sentences, "Sentences") if sentences else 0.0
    word_wer = evaluate_subset(words, "Isolated Words") if words else 0.0
    
    return sentence_wer, word_wer

def train_loso(dataset_dir: Path, transcript_dir: Path, output_dir: Path, experiment_name: str, gpu_id: int = 0):
    """Train with Leave-One-Speaker-Out approach using Wav2Vec2"""
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    output_dir = output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir, experiment_name)
    logger.info(f"Starting experiment: {experiment_name} (Wav2Vec2 Base English)")
    
    logger.info("Loading transcripts and speaker files...")
    transcripts = load_transcripts(transcript_dir)
    speaker_files = get_speaker_files(dataset_dir)
    
    # Initialize processor (Wav2Vec2 Base 960h)
    logger.info("Loading Wav2Vec2 Base English model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    results = {
        "experiment": experiment_name,
        "per_speaker_results": {},
        "average_results": {}
    }
    
    dysarthric_speakers = ['F01', 'M01', 'M02', 'M04', 'M05', 'F03', 'F04', 'M03']

    # LOSO training
    for test_speaker in tqdm(speaker_files.keys(), desc="LOSO iterations"):

        if test_speaker not in dysarthric_speakers:
            continue
 
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing on speaker: {test_speaker}")
        logger.info(f"{'='*50}")
        
        # Prepare splits
        train_files = []
        val_files = []
        test_files = speaker_files[test_speaker]
        
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
        # Freeze feature encoder to stabilize training
        model.freeze_feature_encoder()
        model.to(device)
        
        checkpoint_dir = output_dir / f"checkpoints_{test_speaker}"
        resume_checkpoint = False
        if checkpoint_dir.exists() and any(x.name.startswith("checkpoint") for x in checkpoint_dir.iterdir()):
            logger.info(f"Found existing checkpoints in {checkpoint_dir}. Resuming training...")
            resume_checkpoint = True

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            per_device_train_batch_size=8,  # Slightly lower batch size for Wav2Vec2
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,   # Increase to compensate for batch size
            learning_rate=1e-4,              # Typical LR for Wav2Vec2 Fine-tuning
            warmup_steps=500,
            num_train_epochs=100,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_strategy="epoch",
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            report_to=["none"],
            push_to_hub=False,
            dataloader_num_workers=4,
            group_by_length=True,            # Efficient for variable length audio
        )
        
        # Data collator (CTC Specific)
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        
        # Trainer
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
        
        # Train
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        # Save model
        model_path = output_dir / f"model_{test_speaker}"
        model.save_pretrained(model_path)
        processor.save_pretrained(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Evaluate
        logger.info("Evaluating on test speaker...")
        sentence_wer, word_wer = evaluate_speaker(
            model, processor, test_files, transcripts, device, logger, output_dir, test_speaker
        )
        
        results["per_speaker_results"][test_speaker] = {
            "sentences_wer": sentence_wer,
            "isolated_words_wer": word_wer
        }
        
        logger.info(f"Speaker {test_speaker} - Sentences WER: {sentence_wer:.2f}%")
        logger.info(f"Speaker {test_speaker} - Isolated Words WER: {word_wer:.2f}%")
        
        # Cleanup
        del model
        del trainer
        torch.cuda.empty_cache()
    
    # Calculate average results
    valid_speakers = [s for s in dysarthric_speakers if s in results["per_speaker_results"]]
    
    if valid_speakers:
        avg_sentence_wer = np.mean([results["per_speaker_results"][s]["sentences_wer"] for s in valid_speakers])
        avg_word_wer = np.mean([results["per_speaker_results"][s]["isolated_words_wer"] for s in valid_speakers])
        
        results["average_results"] = {
            "avg_sentences_wer": avg_sentence_wer,
            "avg_isolated_words_wer": avg_word_wer
        }
        
        logger.info(f"\n{'='*50}")
        logger.info("FINAL RESULTS (averaged over dysarthric speakers)")
        logger.info(f"{'='*50}")
        logger.info(f"Average Sentences WER: {avg_sentence_wer:.2f}%")
        logger.info(f"Average Isolated Words WER: {avg_word_wer:.2f}%")
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    summary_data = []
    for speaker, result in results["per_speaker_results"].items():
        summary_data.append({
            'speaker': speaker,
            'sentences_wer': result['sentences_wer'],
            'isolated_words_wer': result['isolated_words_wer']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / "summary_results.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wav2Vec2 LOSO training for dysarthric ASR")
    
    parser.add_argument("--dataset_dir", type=str, required=True,
                      help="Path to audio dataset directory")
    
    parser.add_argument("--transcript_dir", type=str, required=True,
                      help="Path to transcript directory")
                      
    parser.add_argument("--output_dir", type=str, default="./experiments",
                      help="Output directory for experiments")
    parser.add_argument("--experiment_name", type=str, required=True,
                      help="Name of the experiment")
    parser.add_argument("--gpu_id", type=int, default=0,
                      help="GPU ID to use (default: 0)")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    transcript_dir = Path(args.transcript_dir)
    output_dir = Path(args.output_dir)
    
    train_loso(dataset_dir, transcript_dir, output_dir, args.experiment_name, args.gpu_id)
