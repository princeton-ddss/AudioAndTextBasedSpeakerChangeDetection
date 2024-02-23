from torch import device
from typing import Union

from SpeakerChangeDetection.speaker_change_detection_main_function import run_speaker_change_detection_models
from EnsembleSpeakerChangeDetection.ensemble_detection import ensemble_detection


def run_ensemble_audio_text_based_speaker_change_detection_model(detection_models, min_speakers, max_speakers,
                                                           audio_file_input_path, audio_file_input_name,
                                                           transcription_input_path, transcription_file_input_name,
                                                           detection_output_path,  hf_access_token,
                                                           llama2_model_path, pyannote_model_path, device: Union[str, device]=None,
                                                           detection_llama2_output_path=None, temp_output_path=None, ensemble_voting = ['majority', 'unanimity']):
    """The main function to run the ensemble audio-and-text-based speaker change detection model by passing transcription
    files and audio files as inputs
    :param detection_models: A list of names of speaker change detection models to be run
    :type detection_models: list
    :param min_speakers: The minimal number of speakers in the input audio file
    :type: min_speakers: int
    :param max_speakers: The maximal number of speakers in the input audio file
    :type: max_speakers: int
    :param audio_file_input_path: A path which contains an input audio file
    :type audio_file_input_path: str
    :param audio_file_input_name: A audio file name containing the file type
    :type audio_file_input_name: str
    :param transcription_input_path: A path where a transcription output csv file is saved
    :type transcription_input_path: str
    :param transcription_file_input_name: A transcription output csv file name ending with .csv
    :type: transcription_file_input_name: str
    :param detection_output_path: A path to save the speaker change detection output in csv file
    :type detection_output_path: str
    :param hf_access_token: Access token to HuggingFace
    :type hf_access_token: str
    :param llama2_model_path: A path where the Llama2 model files are saved
    :type llama2_model_path: str
    :param pyannote_model_path: A path where the Pyannote model files are saved, default to None
    :type pyannote_model_path: str, optional
    :param device:Device type to run the model, defaults to None so GPU would be automatically
    used if it is available
    :type: str or torch.device, optional
    :param detection_llama2_output_path: A path where the pre-run Llama2 speaker change detection output in csv file
    is saved if exists, default to None
    :type detection_llama2_output_path: str, optional
    :param temp_output_path: A path to save the current run of Llama2 speaker change detection output
    to avoid future rerunning, default to None
    :type temp_output_path: str, optional
    :param ensemble_output_path: A path to save the ensemble detection output in csv file
    :type ensemble_output_path: str
    :param ensemble_voting: A list of voting methods to be used to build the final ensemble model
    :type ensemble_voting: list
    :return: None
    :rtype: None
    """
    run_speaker_change_detection_models(audio_file_input_path, audio_file_input_name, min_speakers, max_speakers,
                                       transcription_input_path, transcription_file_input_name, detection_models,
                                       detection_output_path, hf_access_token,
                                       llama2_model_path, pyannote_model_path,
                                       device,
                                       detection_llama2_output_path, temp_output_path)

    ensemble_detection(detection_output_path, transcription_file_input_name, detection_output_path, ensemble_voting)

