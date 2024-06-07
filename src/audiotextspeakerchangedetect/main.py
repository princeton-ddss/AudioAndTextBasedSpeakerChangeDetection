from torch import device
from typing import Union

from SpeakerChangeDetection.speaker_change_detection_main_function import (
    run_speaker_change_detection_models,
)
from EnsembleSpeakerChangeDetection.ensemble_detection import ensemble_detection


def run_ensemble_audio_text_based_speaker_change_detection_model(
    detection_models: list,
    min_speakers: int,
    max_speakers: int,
    audio_file_input_path: str,
    audio_file_input_name: str,
    transcription_input_path: str,
    transcription_file_input_name: str,
    detection_output_path: str,
    hf_access_token: str,
    llama2_model_path: str,
    pyannote_model_path: str,
    device: Union[str, device] = None,
    detection_llama2_output_path: str = None,
    temp_output_path: str = None,
    ensemble_voting: str = ["majority", "unanimity"],
):
    """The main function to run the ensemble audio-and-text-based speaker change detection model by passing transcription
    files and audio files as inputs
    Args:
    detection_models: A list of speaker change detection models names
    min_speakers: The minimal number of speakers in the input audio file
    max_speakers: The maximal number of speakers in the input audio file
    audio_file_input_path: A path which contains an input audio file
    audio_file_input_name: A audio file name containing the file type
    transcription_input_path: A path where a transcription output csv file
    is saved
    transcription_file_input_name: A transcription output csv file name
    ending with .csv
    detection_output_path: A path to save the speaker change detection
    output in csv file
    hf_access_token: Access token to HuggingFace
    llama2_model_path: A path where the Llama2 model files are saved
    pyannote_model_path: A path where the Pyannote model files are saved
    device:Device type to run the model, defaults to None so GPU would
    be automatically used if it is available
    detection_llama2_output_path: A path where the pre-run Llama2 speaker
    change detection output in csv file is saved if exists, default to None
    temp_output_path: A path to save the current run of Llama2 speaker
    change detection output to avoid future rerunning, default to None
    ensemble_output_path: A path to save the ensemble detection output
    ensemble_voting: A list of voting methods to aggregate predictions

    Returns:
        None
    """
    run_speaker_change_detection_models(
        audio_file_input_path,
        audio_file_input_name,
        min_speakers,
        max_speakers,
        transcription_input_path,
        transcription_file_input_name,
        detection_models,
        detection_output_path,
        hf_access_token,
        llama2_model_path,
        pyannote_model_path,
        device,
        detection_llama2_output_path,
        temp_output_path,
    )

    ensemble_detection(
        detection_output_path,
        transcription_file_input_name,
        detection_output_path,
        ensemble_voting,
    )
