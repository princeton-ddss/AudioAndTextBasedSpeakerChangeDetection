""""
Define global fixture of general function inputs to be used
"""

import os

import pytest


class Inputs:
    def __init__(self):
        self.audio_name = "sample_data_short.wav"
        self.detection_models = ["pyannote", "clustering", "nlp"]
        self.min_speakers = 2
        self.max_speakers = 10
        self.transcription_name = "sample_data_short.csv"
        self.device = None

    def set_llama2_inputs(
        self, llama2_model_path, temp_output_path=None, llama2_output_path=None
    ):
        self.llama2_model_path = llama2_model_path
        if not llama2_output_path:
            self.llama2_output_path = llama2_output_path
        if not temp_output_path:
            self.temp_output_path = temp_output_path


def pytest_addoption(parser):
    parser.addoption(
        "--test_dir",
        action="store",
        default="/Users/jf3375/PycharmProjects/AudioAndTextBasedSpeakerChangeDetection/src/audiotextspeakerchangedetect/tests",
        help="Test Directory",
    )

    parser.addoption(
        "--audio_dir",
        action="store",
        default="/Users/jf3375/PycharmProjects/AudioAndTextBasedSpeakerChangeDetection/src/audiotextspeakerchangedetect/tests/inputs",
        help="Audio Input Directory",
    )

    parser.addoption(
        "--transcription_dir",
        action="store",
        default="/Users/jf3375/PycharmProjects/AudioAndTextBasedSpeakerChangeDetection/src/audiotextspeakerchangedetect/tests/inputs",
        help="Transcription Input Directory",
    )

    parser.addoption(
        "--output_dir",
        action="store",
        default="/Users/jf3375/PycharmProjects/AudioAndTextBasedSpeakerChangeDetection/src/audiotextspeakerchangedetect/tests/outputs",
        help="Test Output Directory",
    )

    parser.addoption(
        "--hf_access_token",
        action="store",
        default="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",
        help="HuggingFace Access Token",
    )

    parser.addoption(
        "--pyannote_model_path",
        action="store",
        default="/Users/jf3375/Dropbox (Princeton)/models/Pyannote3.1/Diarization",
        help="PyAnnote Model Path",
    )

    parser.addoption(
        "--llama2_model_path",
        action="store",
        default="/Users/jf3375/Dropbox (Princeton)/models/llama",
        help="Llama2 Model Path",
    )


@pytest.fixture
def general_inputs(request):
    general_inputs = Inputs()
    general_inputs.audio_dir = request.config.getoption("--audio_dir")
    general_inputs.transcription_dir = request.config.getoption("--transcription_dir")
    general_inputs.output_dir = request.config.getoption("--output_dir")
    general_inputs.hf_access_token = request.config.getoption("--hf_access_token")
    general_inputs.pyannote_model_path = request.config.getoption(
        "--pyannote_model_path"
    )
    return general_inputs


@pytest.fixture
def llama2_inputs(general_inputs, request):
    general_inputs.detection_models = ["llama2-70b"]
    general_inputs.llama2_model_path = request.config.getoption("--llama2_model_path")
    return general_inputs
