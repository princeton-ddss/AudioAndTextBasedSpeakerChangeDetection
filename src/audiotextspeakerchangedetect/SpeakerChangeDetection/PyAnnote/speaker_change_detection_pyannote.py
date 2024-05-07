import os
from pyannote.audio import Pipeline
import torch
from typing import Union


def pyannote_speakerchangedetection(
    audio_file: str,
    min_speakers: int,
    max_speakers: int,
    hf_access_token: str,
    device: Union[str, torch.device] = None,
    pyannote_model_path: str=None,
):
    """..."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if pyannote_model_path:
        print(f"Using Pyannote model: {pyannote_model_path}")
        detection_pipeline = Pipeline.from_pretrained(
            os.path.join(pyannote_model_path, "config.yaml")
        )
    else:
        print("Using default Pyannote model: pyannote/speaker-diarization-3.1")
        detection_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_access_token
        )

    detection_pipeline.to(device)
    detection_result = detection_pipeline(
        audio_file,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    labeled_timestamps = {}
    for turn, _, speaker in detection_result.itertracks(yield_label=True):
        labeled_timestamps[turn.start] = speaker

    return labeled_timestamps
