import pytest

from ..SpeakerChangeDetection.speaker_change_detection_main_function import (
    run_speaker_change_detection_models,
)
from ..EnsembleSpeakerChangeDetection.helpers import map_string_to_bool

"""
To run all tests: pytest xx.py --audio_dir=
To run slow tests: pytest xx.py -m slow
To skip slow tests: pytest -m "not llama2"
"""


@pytest.mark.notllama2
def test_run_speaker_change_detection_models_withoutllama2(general_inputs):
    results_df = run_speaker_change_detection_models(
        audio_dir=general_inputs.audio_dir,
        audio_name=general_inputs.audio_name,
        detection_models=general_inputs.detection_models,
        min_speakers=general_inputs.min_speakers,
        max_speakers=general_inputs.max_speakers,
        transcription_dir=general_inputs.transcription_dir,
        transcription_name=general_inputs.transcription_name,
        output_dir=general_inputs.output_dir,
        hf_access_token=general_inputs.hf_access_token,
        pyannote_model_path=general_inputs.pyannote_model_path,
    )
    assert list(results_df["speaker_change_spectralclustering"]) == [
        True,
        False,
        True,
        True,
        False,
        False,
    ]

    assert list(results_df["speaker_change_pyannote"]) == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]

    assert list(results_df["speaker_change_nlp"]) == [
        "NotSure",
        False,
        True,
        "NotSure",
        "NotSure",
        "NotSure",
    ]


@pytest.mark.llama2
def test_run_speaker_change_detection_models_llama2only(llama2_inputs):
    results_df = run_speaker_change_detection_models(
        audio_dir=llama2_inputs.audio_dir,
        audio_name=llama2_inputs.audio_name,
        detection_models=llama2_inputs.detection_models,
        min_speakers=llama2_inputs.min_speakers,
        max_speakers=llama2_inputs.max_speakers,
        transcription_dir=llama2_inputs.transcription_dir,
        transcription_name=llama2_inputs.transcription_name,
        output_dir=llama2_inputs.output_dir,
        hf_access_token=llama2_inputs.hf_access_token,
        pyannote_model_path=llama2_inputs.pyannote_model_path,
        llama2_model_path=llama2_inputs.llama2_model_path,
    )
    print("Test Llama2")
    # The llama2 output by default is a string
    assert list(results_df["speaker_change_llama2"].apply(map_string_to_bool)) == [
        True,
        False,
        False,
        True,
        False,
        True,
    ]
    print("Done")
