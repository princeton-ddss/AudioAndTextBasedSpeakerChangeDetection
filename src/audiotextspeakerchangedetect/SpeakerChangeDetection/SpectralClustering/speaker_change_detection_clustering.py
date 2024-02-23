from typing import Union
import torch
import os
from resemblyzer import preprocess_wav, VoiceEncoder, sampling_rate
from spectralcluster import SpectralClusterer

# Create Dict to Map Time Segment to Speaker
def create_labelling(labels, wav_splits):
    times = [((s.start + s.stop) / 2) / sampling_rate for s in wav_splits]
    start_time = 0

    timestamp_speaker = {}

    for i, time in enumerate(times):
        if i > 0 and labels[i] != labels[i - 1]:
            timestamp_speaker[start_time] = 'speaker' + str(labels[i - 1])
            start_time = time
        if i == len(times) - 1:
            timestamp_speaker[start_time] = 'speaker' + str(labels[i])
    return timestamp_speaker

def spectralclustering_speakerchangedetection(audio_file_input_path, audio_file_input_name,
                                           min_speakers, max_speakers, device: Union[str, torch.device]=None):
    wav = preprocess_wav(os.path.join(audio_file_input_path, audio_file_input_name))

    encoder = VoiceEncoder(device)
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)

    # Clustering of Speakers based on Embedding
    clusterer = SpectralClusterer(
        min_clusters=min_speakers,
        max_clusters=max_speakers)

    labels = clusterer.predict(cont_embeds)
    timestamp_speaker = create_labelling(labels, wav_splits)
    return timestamp_speaker



