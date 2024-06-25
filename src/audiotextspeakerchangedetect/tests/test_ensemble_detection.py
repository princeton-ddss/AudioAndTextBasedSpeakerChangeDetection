import os
import pytest
from ..EnsembleSpeakerChangeDetection.ensemble_detection import (
    ensemble_detection,
)

"""
To run the test of ensemble function:
cd <tests_main_dir>
pytest .. --test_dir=$pwd -m ensemble
"""


@pytest.mark.ensemble
def test_ensemble_detection(request):
    tests_dir = request.config.getoption("--test_dir")
    detection_fpath = os.path.join(tests_dir, "inputs")
    detection_fname = "sample_data_short_ensemble.csv"
    ensemble_opath = os.path.join(tests_dir, "outputs")
    ensemble_voting = ["majority", "singularity"]

    ensemble_df = ensemble_detection(
        detection_fpath, detection_fname, ensemble_opath, ensemble_voting
    )

    print("Test Singularity Method")
    assert list(ensemble_df["speaker_change_ensemble_singularity"]) == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    print("Done")

    print("Test Majority Method")
    assert list(ensemble_df["speaker_change_ensemble_majority"]) == [
        True,
        False,
        True,
        True,
        False,
        False,
    ]
    print("Done")
