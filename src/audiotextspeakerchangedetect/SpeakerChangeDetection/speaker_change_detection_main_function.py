from typing import Union
import pandas as pd
import torch
import os
import spacy

from .SpectralClustering.speaker_change_detection_clustering import (
    spectralclustering_speakerchangedetection,
)
from .PyAnnote.speaker_change_detection_pyannote import pyannote_speakerchangedetection
from .NLP.speaker_change_detection_nlp import nlp_speakerchangedetection
from .Llama2.speaker_change_detection_llama2 import llama2_speakerchangedetection

from .helpers import merge_detection_audio_results_with_transcription


def run_speaker_change_detection_models(
    audio_file: str,
    detection_models: list[str],
    min_speakers: int,
    max_speakers: int,
    transcription_file: str,
    output_dir: str,
    hf_access_token: str,
    llama2_model_path: str = None,
    llama2_output: str = None,
    pyannote_model_path: str = None,
    device: Union[str, torch.device] = None,
    tmp_dir: str = None,
):
    """Run speaker change detection models on an input audio file.

    Detailed explaination here...

    Args:
        audio_file: A path to an input audio file.
        detection_models: A list of speaker change detection models to run.
        min_speakers: The minimal number of speakers in the input audio file.
        max_speakers: The maximal number of speakers in the input audio file.
        transcription_input_path: A path to a Whisper transcription CSV file.
        output_dir: A path to save the results of speaker change detection models.
        hf_access_token: Access token to HuggingFace.
        llama2_model_path: A path where the Llama2 model files are saved
        pyannote_model_path: The path where Pyannote model files are stored (default: None).
        llama2_output: A path to pre-computed Llama2 speaker change detection results.
        device: The device to run the models on. Default `None` uses GPU, if available.
        tmp_dir: A path to save the current run of Llama2 speaker change detection results.

    Returns:
        None
    """

    llama2_models = [x for x in detection_models if x.startswith("llama2")]
    if len(llama2_models) > 1:
        raise Exception(
            f"Should select at most one llama2 model for detection ({len(llama2_models)} provided)."
        )

    # Set max_split_size_mb to reduce memory fregmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    # TODO: A comment about why this is needed...
    torch.cuda.empty_cache()

    whisper_df = pd.read_csv(transcription_file)

    if "clustering" in detection_models:
        print("Running clustering-based speaker detection...")
        timestamp_speaker_clustering = spectralclustering_speakerchangedetection(
            audio_file,
            min_speakers,
            max_speakers,
            "cpu",  # use CPU to avoid poor GPU utilization
        )
        print("Done!")
        speakers_clustering, speaker_change_clustering = (
            merge_detection_audio_results_with_transcription(
                timestamp_speaker_clustering, whisper_df
            )
        )
        whisper_df["speaker_spectralclustering"] = speakers_clustering
        whisper_df["speaker_change_spectralclustering"] = speaker_change_clustering
    if "pyannote" in detection_models:
        print("Running Pyannote speaker detection...")
        timestamp_speaker_pyannote = pyannote_speakerchangedetection(
            audio_file,
            min_speakers,
            max_speakers,
            hf_access_token,
            device,
            pyannote_model_path,
        )
        print("Done!")
        speakers_pyannote, speaker_change_pyannote = (
            merge_detection_audio_results_with_transcription(
                timestamp_speaker_pyannote, whisper_df
            )
        )
        whisper_df["speaker_pyannote"] = speakers_pyannote
        whisper_df["speaker_change_pyannote"] = speaker_change_pyannote
    if "nlp" in detection_models:
        print("Running NLP-based speaker detection...")
        nlp_model = spacy.load("en_core_web_lg")
        whisper_df["speaker_change_nlp"] = nlp_speakerchangedetection(
            whisper_df, nlp_model
        )
        print("Done!")
    if llama2_models:
        llama2_model = llama2_models[0]
        # TODO: validate llama2_model string
        llama_model_size = llama2_model.split("-")[-1]
        if llama2_output:
            print(f"Using provided Llama2 output: {llama2_output}")
            df_llama2 = pd.read_csv(llama2_output)
        else:
            print("Running LLama2-based speaker change detection...")
            # TODO: check that llama2_model_path is not None
            df_llama2 = llama2_speakerchangedetection(
                whisper_df, llama2_model_path, llama_model_size
            )
            print("Done!")
            if tmp_dir:
                print(f"Writing Llama2 results to {tmp_dir}/{transcription_file}")
                df_llama2.to_csv(os.path.join(tmp_dir, transcription_file), index=False)

        df_llama2 = df_llama2.drop_duplicates(subset=["segmentid"])
        df_llama2 = df_llama2.drop(columns="text")
        df_llama2["segmentid"] = df_llama2["segmentid"].astype(int)
        whisper_df = pd.merge(whisper_df, df_llama2, on="segmentid", how="left")

    print(f"Writing results to {output_dir}/{transcription_file}")
    whisper_df.to_csv(os.path.join(output_dir, transcription_file), index=False)
    return whisper_df
