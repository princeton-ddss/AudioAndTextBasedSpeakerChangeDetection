import os
from pyannote.audio import Pipeline
import torch
from typing import Union

def pyannote_speakerchangedetection(audio_file_input_path, audio_file_input_name, min_speakers, max_speakers, hf_access_token,
                                 device: Union[str, torch.device]=None, pyannote_model_path=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if pyannote_model_path:
        detection_pipeline = Pipeline.from_pretrained(os.path.join(pyannote_model_path, 'config.yaml'))
    else:
        detection_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_access_token)

    detection_pipeline.to(device)

    detection_result = detection_pipeline(os.path.join(audio_file_input_path, audio_file_input_name),
                                                min_speakers = min_speakers, max_speakers = max_speakers)

    timestamp_speaker = {}
    for turn, _, speaker in detection_result.itertracks(yield_label=True):
        timestamp_speaker[turn.start] = speaker

    return timestamp_speaker




