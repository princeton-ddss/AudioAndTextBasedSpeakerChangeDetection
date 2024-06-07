from typing import Union
import torch
from resemblyzer import preprocess_wav, VoiceEncoder, sampling_rate
from spectralcluster import SpectralClusterer


def create_labelling(labels, wav_splits):
    """Create a dict to map time segments to speakers."""
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    start_time = 0
    timestamp_speaker = {}
    for i, time in enumerate(times):
        if i > 0 and labels[i] != labels[i - 1]:
            timestamp_speaker[start_time] = "speaker" + str(labels[i - 1])
            start_time = time
        if i == len(times) - 1:
            timestamp_speaker[start_time] = "speaker" + str(labels[i])
    return timestamp_speaker


def spectralclustering_speakerchangedetection(
    audio_file:str,
    min_speakers:int,
    max_speakers:int,
    device:Union[str, torch.device]
):
    """
    The function to run speaker change detection using spectral clustering

    Args:
        audio_file: the audio file name
        min_speakers: the minimum number of speakers in the audio file
        max_speakers: the maximum number of speakers in the audio file
        device: device to run the models

    Returns:
        labeled_timestamps: the dictionary with timestamps as keys and their corresponding speakers as values
    """

    wav = preprocess_wav(audio_file)
    encoder = VoiceEncoder(device)
    _, cont_embeds, wav_splits = encoder.embed_utterance(
        wav, return_partials=True, rate=16
    )
    clusterer = SpectralClusterer(min_clusters=min_speakers, max_clusters=max_speakers)
    labels = clusterer.predict(cont_embeds)
    timestamp_speakers = create_labelling(labels, wav_splits)
    return timestamp_speakers
