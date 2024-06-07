from bisect import bisect_left
from typing import TypeVar


def merge_detection_audio_results_with_transcription(
    timestamp_speaker: dict,
    whisper_df:TypeVar('pandas.core.frame.DataFrame'),
):
    """Merge speaker change detection results with transcription results.

    Args:
        timestamp_speaker: the dictionary with timestamp as keys and speaker ids as values
        whisper_df: the pandas dataframe of whisper transcription results

    Returns:
        speakers: the list of speakers for each transcription part
        speaker_changes: the list of true or false to indicate if the speaker changes at each transcription part
    """

    timestamps = list(timestamp_speaker.keys())
    speakers = [None] * whisper_df.shape[0]
    speaker_changes = [None] * whisper_df.shape[0]
    prev_speaker, curr_speaker = None, None
    for idx, segment in whisper_df.iterrows():
        timestamp_idx = max(
            bisect_left(timestamps, segment["start"]) - 1, 0
        )  # adjust for missing zero timestamp
        curr_speaker = timestamp_speaker[timestamps[timestamp_idx]]
        speakers[idx] = curr_speaker
        if curr_speaker != prev_speaker:
            speaker_changes[idx] = True
        else:
            speaker_changes[idx] = False
        prev_speaker = curr_speaker
    return speakers, speaker_changes
