"""Run speaker change detection *with* existing Llama2 output."""

from audiotextspeakerchangedetect.main import (
    run_ensemble_audio_text_based_speaker_change_detection_model,
)

detection_models = ["pyannote", "clustering", "nlp", "llama2-70b"]
min_speakers = 2
max_speakers = 10
audio_file = "/scratch/gpfs/jf3375/modern_family/audio/sample_data/sample_data.wav"
transcription_file = "/scratch/gpfs/jf3375/modern_family/output/sample_data.csv"
detection_output_path = "/scratch/gpfs/jf3375/modern_family/output/detection"
hf_access_token = "<hf_access_token>"
pyannote_model_path = "/scratch/gpfs/jf3375/models/pyannote3.1/Diarization"
detection_llama2_output_path = "/scratch/gpfs/jf3375/modern_family/output/detection/Llama2/70b"  # Existing llama2 output
tmp_dir = "/scratch/gpfs/jf3375/modern_family/temp"
ensemble_voting = ["majority", "unanimity"]

run_ensemble_audio_text_based_speaker_change_detection_model(
    detection_models,
    min_speakers,
    max_speakers,
    audio_file,
    transcription_file,
    detection_output_path,
    hf_access_token=hf_access_token,
    pyannote_model_path=pyannote_model_path,
    detection_llama2_output_path,
    tmp_dir=tmp_dir,
    ensemble_voting=ensemble_voting,
)
