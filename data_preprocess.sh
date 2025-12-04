#TTS data preprocessing
poetry run python scripts/preprocess_speech_data.py 16000 LJSpeech-1.1 /processed/dataset/

#SSL Features Extraction
poetry run python scripts/extract_dataset_embeddings.py wavlm /processed/dataset/ wavlm_feats/

#Train Urhythmic Segmenter
poetry run python recipes/train_urhythmic_segmenter.py wavlm_feats checkpoints/segmenter.pth 3

#Train Urhythmic Rhythm Model
poetry run python recipes/train_urhythmic_rhythm_model.py speaker_id global wavlm_feats checkpoints/segmenter.pth output_urhythmic_global

poetry run python recipes/train_urhythmic_rhythm_model.py speaker_id fine wavlm_feats checkpoints/segmenter.pth output_urhythmic_fine

#Train Syllable Rhythm model
#poetry run python recipes/train_syllable_rhythm_model.py speaker_id /path/to/speaker/audio checkpoints/segmenter.pth /path/to/save/output


