"""Run speaker change detection without existing Llama2 output."""

from audiotextspeakerchangedetect.main import (
    run_ensemble_audio_text_based_speaker_change_detection_model,
)

detection_models = ["pyannote", "clustering", "nlp", "llama2-70b"]
min_speakers = 2
max_speakers = 10
audio_file = "/scratch/gpfs/jf3375/test/input/bvyvm.wav"
transcription_file = "/scratch/gpfs/jf3375/test/input/bvyvm.csv"
detection_output_path = "/scratch/gpfs/jf3375/test/output"
hf_access_token = "secret"
llama2_model_path = "/scratch/gpfs/jf3375/models/llama"
pyannote_model_path = "/scratch/gpfs/jf3375/models/pyannote3.1/Diarization"
tmp_dir = "/scratch/gpfs/jf3375/test/temp"
ensemble_voting = ["majority", "unanimity"]

run_ensemble_audio_text_based_speaker_change_detection_model(
    detection_models,
    min_speakers,
    max_speakers,
    audio_file,
    transcription_file,
    detection_output_path,
    hf_access_token=hf_access_token,
    llama2_model_path,
    pyannote_model_path,
    tmp_dir=tmp_dir,
    ensemble_voting=ensemble_voting,
)
