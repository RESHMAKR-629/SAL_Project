from pathlib import Path
import librosa
from rnv.converter import Converter
from rnv.ssl.models import WavLM
from rnv.utils import get_vocoder_checkpoint_path
from tqdm import tqdm
import torch
import shutil
import sys
from datetime import datetime
import os

CHECKPOINTS_DIR = "/asr4/reshma/SAL_pro/RnV/checkpoints"
MAIN_PATH = Path("/asr4/reshma/SAL_pro/RnV/processed/torgo_processed")
OUTPUT_BASE = Path("/asr4/reshma/SAL_pro/RnV/experiments")
RHYTHM_MODELS_SYLLABLE_DIR = Path("/asr4/reshma/SAL_pro/RnV/output_syllable_rhythm")
RHYTHM_MODELS_URHYTHMIC_FINE_DIR = Path("/asr4/reshma/SAL_pro/RnV/output_urhythmic_fine")
RHYTHM_MODELS_URHYTHMIC_GLOBAL_DIR = Path("/asr4/reshma/SAL_pro/RnV/output_urhythmic_global")

TARGET_STYLE_FEATS_PATH = "/asr4/reshma/SAL_pro/RnV/wavlm_lj_emb-wavlm"
TARGET_RHYTHM_MODEL_URHYTHMIC_FINE = Path("/asr4/reshma/SAL_pro/RnV/output_urhythmic_fine_LJ/LJ_Speakers_fine_urhythmic_model.pth")
TARGET_RHYTHM_MODEL_URHYTHMIC_GLOBAL = Path("/asr4/reshma/SAL_pro/RnV/output_urhythmic_global_LJ/LJ_Speakers_global_urhythmic_model.pth")
TARGET_RHYTHM_MODEL_SYLLABLE_FINE = Path("/asr4/reshma/SAL_pro/RnV/output_syllable_rhythm_lj/LJ_speakers_syllable_models.pth")
TARGET_RHYTHM_MODEL_SYLLABLE_GLOBAL = Path("/asr4/reshma/SAL_pro/RnV/output_syllable_rhythm_lj/LJ_speakers_syllable_models.pth")

EXPERIMENTS = {
    "Original": {"rhythm": None, "voice": None, "rhythm_type": None},
    "Vocoded": {"rhythm": None, "voice": None, "rhythm_type": None, "vocoded": True},
    "kNN-VC": {"rhythm": None, "voice": True, "rhythm_type": None},
    "Urhythmic_Fine": {"rhythm": "urhythmic", "voice": None, "rhythm_type": "fine"},
    "Urhythmic_Global": {"rhythm": "urhythmic", "voice": None, "rhythm_type": "global"},
    "Syllable_Fine": {"rhythm": "syllable", "voice": None, "rhythm_type": "fine"},
    "Syllable_Global": {"rhythm": "syllable", "voice": None, "rhythm_type": "global"},
    "Urhythmic_Fine_kNN-VC": {"rhythm": "urhythmic", "voice": True, "rhythm_type": "fine"},
    "Urhythmic_Global_kNN-VC": {"rhythm": "urhythmic", "voice": True, "rhythm_type": "global"},
    "Syllable_Fine_kNN-VC": {"rhythm": "syllable", "voice": True, "rhythm_type": "fine"},
    "Syllable_Global_kNN-VC": {"rhythm": "syllable", "voice": True, "rhythm_type": "global"},
}

# Setup logging
log_file = OUTPUT_BASE / f"conversion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

logger = Logger(log_file)
sys.stdout = logger

print(f"Conversion started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {log_file}")
print("="*60)

vocoder_checkpoint_path = get_vocoder_checkpoint_path(CHECKPOINTS_DIR)
feature_extractor = WavLM()
segmenter_path = Path(f"{CHECKPOINTS_DIR}/segmenter.pth")

if not Path(TARGET_STYLE_FEATS_PATH).exists():
    print(f"ERROR: Target features path not found: {TARGET_STYLE_FEATS_PATH}")
    sys.exit(1)

print(f"Target features path: {TARGET_STYLE_FEATS_PATH}")
print(f"Segmenter path: {segmenter_path}")
print(f"Syllable rhythm models dir: {RHYTHM_MODELS_SYLLABLE_DIR}")
print(f"Urhythmic fine models dir: {RHYTHM_MODELS_URHYTHMIC_FINE_DIR}")
print(f"Urhythmic global models dir: {RHYTHM_MODELS_URHYTHMIC_GLOBAL_DIR}")

subdirs = ["F_Con", "F_Dys", "M_Con", "M_Dys"]

for subdir in subdirs:
    subdir_path = MAIN_PATH / subdir
    if not subdir_path.exists():
        print(f"Skipping {subdir} - directory not found")
        continue
    
    speaker_dirs = sorted([d for d in subdir_path.iterdir() if d.is_dir()])
    
    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        print(f"\n{'='*60}")
        print(f"Processing speaker: {speaker_id} from {subdir}")
        print(f"{'='*60}")
        
        # Find all .wav files (case-insensitive) in all subdirectories
        wav_files = sorted([f for f in speaker_dir.rglob("*") if f.suffix.lower() == ".wav"])
        
        if not wav_files:
            print(f"No wav files found for {speaker_id}")
            continue
        
        print(f"Found {len(wav_files)} wav files in {speaker_dir}")
        
        source_rhythm_models = {
            'urhythmic_fine': RHYTHM_MODELS_URHYTHMIC_FINE_DIR / f"{speaker_id}_fine_urhythmic_model.pth",
            'urhythmic_global': RHYTHM_MODELS_URHYTHMIC_GLOBAL_DIR / f"{speaker_id}_global_urhythmic_model.pth",
            'syllable_fine': RHYTHM_MODELS_SYLLABLE_DIR / f"{speaker_id}_syllable_fine_models.pth",
            'syllable_global': RHYTHM_MODELS_SYLLABLE_DIR / f"{speaker_id}_syllable_global_models.pth",
        }
        
        if not source_rhythm_models['syllable_fine'].exists():
            alt_path = RHYTHM_MODELS_SYLLABLE_DIR / f"{speaker_id}_syllable_models.pth"
            if alt_path.exists():
                source_rhythm_models['syllable_fine'] = alt_path
                source_rhythm_models['syllable_global'] = alt_path
        
        models_status = {}
        for key, path in source_rhythm_models.items():
            exists = path.exists()
            models_status[key] = exists
            if not exists:
                print(f"Warning: {key} model not found: {path}")
            else:
                print(f"Found {key} model: {path}")
        
        for exp_name, exp_config in EXPERIMENTS.items():
            print(f"\n  Experiment: {exp_name}")
            
            source_rhythm = None
            target_rhythm = None
            skip_experiment = False
            
            if exp_config["rhythm"]:
                rhythm_type = exp_config["rhythm"]
                model_type = exp_config["rhythm_type"]
                model_key = f"{rhythm_type}_{model_type}"
                
                if not models_status.get(model_key, False):
                    print(f"    Skipping - {model_key} model not found")
                    skip_experiment = True
                else:
                    source_rhythm = source_rhythm_models[model_key]
                    
                    if rhythm_type == "urhythmic":
                        target_rhythm = TARGET_RHYTHM_MODEL_URHYTHMIC_FINE if model_type == "fine" else TARGET_RHYTHM_MODEL_URHYTHMIC_GLOBAL
                    else:
                        target_rhythm = TARGET_RHYTHM_MODEL_SYLLABLE_FINE if model_type == "fine" else TARGET_RHYTHM_MODEL_SYLLABLE_GLOBAL
                    
                    if not target_rhythm.exists():
                        print(f"    Warning: Target rhythm model not found: {target_rhythm}")
                        alt_target = Path("/asr4/reshma/SAL_pro/RnV/output_syllable_rhythm_lj/LJ_speakers_syllable_models.pth")
                        if rhythm_type == "syllable" and alt_target.exists():
                            target_rhythm = alt_target
                            print(f"    Using alternative target: {target_rhythm}")
                        else:
                            print(f"    Skipping - Target rhythm model not found")
                            skip_experiment = True
            
            if skip_experiment:
                continue
            
            output_dir = OUTPUT_BASE / exp_name / subdir / speaker_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            converter = Converter(vocoder_checkpoint_path) if not exp_config["rhythm"] else Converter(
                vocoder_checkpoint_path,
                rhythm_converter=exp_config["rhythm"],
                rhythm_model_type=exp_config["rhythm_type"]
            )
            
            for wav_file in tqdm(wav_files, desc=f"    Processing"):
                try:
                    output_path = output_dir / wav_file.name
                    
                    if exp_name == "Original":
                        shutil.copy(wav_file, output_path)
                        continue
                    
                    source_wav, sr = librosa.load(str(wav_file), sr=None)
                    source_feats = feature_extractor.extract_framewise_features(str(wav_file), output_layer=None).cpu()
                    
                    voice_path = TARGET_STYLE_FEATS_PATH if exp_config["voice"] else None
                    knnvc_topk = 8 if exp_config["voice"] else None
                    lambda_rate = 1.0 if exp_config["voice"] else None
                    
                    if exp_config.get("vocoded", False):
                        converter.convert(source_feats, None, None, None, segmenter_path, save_path=str(output_path))
                    else:
                        converter.convert(
                            source_feats,
                            voice_path,
                            source_rhythm,
                            target_rhythm,
                            segmenter_path,
                            knnvc_topk,
                            lambda_rate,
                            source_wav=source_wav,
                            save_path=str(output_path)
                        )
                    
                except Exception as e:
                    print(f"\n    Error processing {wav_file.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            del converter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

print("\n" + "="*60)
print(f"All experiments completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

logger.close()
sys.stdout = sys.__stdout__
print(f"Log saved to: {log_file}")

