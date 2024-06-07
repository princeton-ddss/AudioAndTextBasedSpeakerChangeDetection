import os
from pyannote.audio import Pipeline
import torch
from typing import Union


def pyannote_speakerchangedetection(
    audio_fdir: str,
    audio_fname: str,
    min_speakers: int,
    max_speakers: int,
    hf_access_token: str,
    device: Union[str, torch.device] = None,
    pyannote_model_path: str=None,
):
    """
    Run Speaker Change Detection using PyAnnote

    Args:
        audio_fdir: input audio file directory,
        audio_fname: input audio file name,
        min_speakers: the number of minimal speakers in audio,
        max_speakers: the number of maximal speakers in audio,
        hf_access_token: huggingface access token,
        device: Union[str, torch.device]: default device to run the model,
        pyannote_model_path: pyannote model path

    Returns:
        labeled_timestamps: the dictionary with timestamps as keys and their corresponding speakers as values
    """

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
        os.path.join(audio_fdir, audio_fname),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    timestamps_speakers = {}
    for turn, _, speaker in detection_result.itertracks(yield_label=True):
        timestamps_speakers[turn.start] = speaker

    return timestamps_speakers
