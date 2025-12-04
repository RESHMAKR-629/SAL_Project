from pathlib import Path
import librosa
import soundfile as sf
from rnv.converter import Converter
from rnv.ssl.models import WavLM
from rnv.utils import get_vocoder_checkpoint_path
import os

# Configuration
CHECKPOINTS_DIR = "/asr4/reshma/SAL_pro/RnV/checkpoints"
vocoder_checkpoint_path = get_vocoder_checkpoint_path(CHECKPOINTS_DIR)
feature_extractor = WavLM()
segmenter_path = Path("/asr4/reshma/SAL_pro/RnV/checkpoints/segmenter.pth")

# Paths
source_wav_path = "/asr4/reshma/SAL_pro/RnV/processed/torgo/F_Dys/F01/wav_arrayMic_F01/wav_arrayMic_F01_0001.wav"
target_style_feats_path = "/asr4/reshma/SAL_pro/RnV/wavlm_torgo_feats/F_Con/FC01/wav_arrayMic_FC01S01-wavlm/"
output_base_dir = "/asr4/reshma/SAL_pro/RnV/output_experiments"

# Create output directory
os.makedirs(output_base_dir, exist_ok=True)

# Load source audio
source_wav, sr = librosa.load(source_wav_path, sr=None)
source_feats = feature_extractor.extract_framewise_features(source_wav_path, output_layer=None).cpu()

# Rhythm models
urhythmic_source = Path("/asr4/reshma/SAL_pro/RnV/output_urhythmic_rhythm/F01_urhythmic_models.pth")
urhythmic_target = Path("/asr4/reshma/SAL_pro/RnV/output_urhythmic_rhythm/FC01_urhythmic_models.pth")
syllable_source = Path("/asr4/reshma/SAL_pro/RnV/output_syllable_rhythm/F01_syllable_models.pth")
syllable_target = Path("/asr4/reshma/SAL_pro/RnV/output_syllable_rhythm/FC01_syllable_models.pth")

# Experiment parameters
knnvc_topk = 4
lambda_rate = 1.0

def save_original_audio():
    """Save original unmodified audio"""
    output_path = os.path.join(output_base_dir, "01_original.wav")
    sf.write(output_path, source_wav, sr)
    print(f"Saved: {output_path}")

def run_vocoded():
    """Vocoded (encoded and decoded without modification)"""
    converter = Converter(vocoder_checkpoint_path, rhythm_converter=None)
    output_path = os.path.join(output_base_dir, "02_vocoded.wav")
    converter.convert(source_feats, None, None, None, segmenter_path, 
                     source_wav=source_wav, save_path=output_path)
    print(f"Saved: {output_path}")

def run_knn_vc():
    """Voice conversion only using kNN-VC"""
    converter = Converter(vocoder_checkpoint_path, rhythm_converter=None)
    output_path = os.path.join(output_base_dir, "03_kNN-VC.wav")
    converter.convert(source_feats, target_style_feats_path, None, None, segmenter_path,
                     knnvc_topk, lambda_rate, save_path=output_path)
    print(f"Saved: {output_path}")

def run_urhythmic_fine():
    """Urhythmic fine-grained rhythm conversion only"""
    converter = Converter(vocoder_checkpoint_path, rhythm_converter="urhythmic", rhythm_model_type="fine")
    output_path = os.path.join(output_base_dir, "04_urhythmic_fine.wav")
    converter.convert(source_feats, None, urhythmic_source, urhythmic_target, segmenter_path,
                     source_wav=source_wav, save_path=output_path)
    print(f"Saved: {output_path}")

def run_urhythmic_global():
    """Urhythmic global rhythm conversion only"""
    converter = Converter(vocoder_checkpoint_path, rhythm_converter="urhythmic", rhythm_model_type="global")
    output_path = os.path.join(output_base_dir, "05_urhythmic_global.wav")
    converter.convert(source_feats, None, urhythmic_source, urhythmic_target, segmenter_path,
                     source_wav=source_wav, save_path=output_path)
    print(f"Saved: {output_path}")

def run_syllable_fine():
    """Syllable fine-grained rhythm conversion only"""
    converter = Converter(vocoder_checkpoint_path, rhythm_converter="syllable", rhythm_model_type="fine")
    output_path = os.path.join(output_base_dir, "06_syllable_fine.wav")
    converter.convert(source_feats, None, syllable_source, syllable_target, segmenter_path,
                     source_wav=source_wav, save_path=output_path)
    print(f"Saved: {output_path}")

def run_syllable_global():
    """Syllable global rhythm conversion only"""
    converter = Converter(vocoder_checkpoint_path, rhythm_converter="syllable", rhythm_model_type="global")
    output_path = os.path.join(output_base_dir, "07_syllable_global.wav")
    converter.convert(source_feats, None, syllable_source, syllable_target, segmenter_path,
                     source_wav=source_wav, save_path=output_path)
    print(f"Saved: {output_path}")

def run_urhythmic_fine_knnvc():
    """Urhythmic fine + kNN-VC"""
    converter = Converter(vocoder_checkpoint_path, rhythm_converter="urhythmic", rhythm_model_type="fine")
    output_path = os.path.join(output_base_dir, "08_urhythmic_fine_kNN-VC.wav")
    converter.convert(source_feats, target_style_feats_path, urhythmic_source, urhythmic_target, 
                     segmenter_path, knnvc_topk, lambda_rate, source_wav=source_wav, save_path=output_path)
    print(f"Saved: {output_path}")

def run_urhythmic_global_knnvc():
    """Urhythmic global + kNN-VC"""
    converter = Converter(vocoder_checkpoint_path, rhythm_converter="urhythmic", rhythm_model_type="global")
    output_path = os.path.join(output_base_dir, "09_urhythmic_global_kNN-VC.wav")
    converter.convert(source_feats, target_style_feats_path, urhythmic_source, urhythmic_target, 
                     segmenter_path, knnvc_topk, lambda_rate, source_wav=source_wav, save_path=output_path)
    print(f"Saved: {output_path}")

def run_syllable_fine_knnvc():
    """Syllable fine + kNN-VC"""
    converter = Converter(vocoder_checkpoint_path, rhythm_converter="syllable", rhythm_model_type="fine")
    output_path = os.path.join(output_base_dir, "10_syllable_fine_kNN-VC.wav")
    converter.convert(source_feats, target_style_feats_path, syllable_source, syllable_target, 
                     segmenter_path, knnvc_topk, lambda_rate, source_wav=source_wav, save_path=output_path)
    print(f"Saved: {output_path}")

def run_syllable_global_knnvc():
    """Syllable global + kNN-VC"""
    converter = Converter(vocoder_checkpoint_path, rhythm_converter="syllable", rhythm_model_type="global")
    output_path = os.path.join(output_base_dir, "11_syllable_global_kNN-VC.wav")
    converter.convert(source_feats, target_style_feats_path, syllable_source, syllable_target, 
                     segmenter_path, knnvc_topk, lambda_rate, source_wav=source_wav, save_path=output_path)
    print(f"Saved: {output_path}")

def main():
    """Run all experiments sequentially"""
    print("Starting audio conversion experiments...")
    
    experiments = [
        save_original_audio,
        run_vocoded,
        run_knn_vc,
        run_urhythmic_fine,
        run_urhythmic_global,
        run_syllable_fine,
        run_syllable_global,
        run_urhythmic_fine_knnvc,
        run_urhythmic_global_knnvc,
        run_syllable_fine_knnvc,
        run_syllable_global_knnvc
    ]
    
    for i, experiment in enumerate(experiments, 1):
        try:
            print(f"\n[{i}/{len(experiments)}] Running {experiment.__name__}...")
            experiment()
        except Exception as e:
            print(f"Error in {experiment.__name__}: {e}")
            continue
    
    print(f"\nAll experiments completed! Output files saved in: {output_base_dir}")
    
    # List all generated files
    print("\nGenerated audio files:")
    for file in sorted(os.listdir(output_base_dir)):
        if file.endswith('.wav'):
            file_path = os.path.join(output_base_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f"  - {file} ({file_size:.1f} KB)")

if __name__ == "__main__":
    main()
    
# from pathlib import Path

# import librosa

# from rnv.converter import Converter
# from rnv.ssl.models import WavLM
# from rnv.utils import get_vocoder_checkpoint_path

# CHECKPOINTS_DIR = "/asr4/reshma/SAL_pro/RnV/checkpoints"
# vocoder_checkpoint_path = get_vocoder_checkpoint_path(CHECKPOINTS_DIR)

# # Initialize the converter with the vocoder checkpoint and rhythm conversion settings
# # You can choose between "urhythmic" or "syllable" for rhythm_converter
# # and "global" or "fine" for rhythm_model_type
# converter = Converter(vocoder_checkpoint_path, rhythm_converter="syllable", rhythm_model_type="global") # or "fine" for fine-grained rhythm conversion

# feature_extractor = WavLM()
# segmenter_path = Path("/asr4/reshma/SAL_pro/RnV/checkpoints/segmenter.pth")

# # Load wav and extract features
# source_wav_path = "/asr4/reshma/SAL_pro/RnV/processed/torgo/F_Dys/F01/wav_arrayMic_F01/wav_arrayMic_F01_0001.wav"
# source_wav, sr = librosa.load(source_wav_path, sr=None)
# source_feats = feature_extractor.extract_framewise_features(source_wav_path, output_layer=None).cpu()

# # Rhythm and Voice Conversion
# target_style_feats_path = "/asr4/reshma/SAL_pro/RnV/wavlm_torgo_feats/F_Con/FC01/wav_arrayMic_FC01S01-wavlm/"
# knnvc_topk = 4
# lambda_rate = 1.
# source_rhythm_model = Path("/asr4/reshma/SAL_pro/RnV/output_syllable_rhythm/F01_syllable_models.pth") # ensure these correspond to the chosen rhythm model type
# target_rhythm_model = Path("/asr4/reshma/SAL_pro/RnV/output_syllable_rhythm/FC01_syllable_models.pth")
# wav = converter.convert(source_feats, target_style_feats_path, source_rhythm_model, target_rhythm_model, segmenter_path, knnvc_topk, lambda_rate, source_wav=source_wav)

# ## or to write the output directly to a file
# output_path = "/asr4/reshma/SAL_pro/RnV/output/output_rnv.wav"
# converter.convert(source_feats, target_style_feats_path, source_rhythm_model, target_rhythm_model, segmenter_path, knnvc_topk, lambda_rate, source_wav=source_wav, save_path=output_path)

# # Rhythm Conversion Only
# output_path = "/asr4/reshma/SAL_pro/RnV/output/output_rhythm_only.wav"
# converter.convert(source_feats, None, source_rhythm_model, target_rhythm_model, segmenter_path, source_wav=source_wav, save_path=output_path)

# # Voice Conversion Only
# output_path = "/asr4/reshma/SAL_pro/RnV/output/output_voice_only.wav"
# converter.convert(source_feats, target_style_feats_path, None, None, segmenter_path, knnvc_topk, lambda_rate, save_path=output_path)