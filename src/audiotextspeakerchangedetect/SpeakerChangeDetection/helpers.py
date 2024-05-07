from bisect import bisect_left


def merge_detection_audio_results_with_transcription(
    # TODO: add type hints
    timestamp_speaker,
    whisper_df,
):
    """Append speaker change detection results with transcription results.

    Args:
        timestamp_speaker: ...
        whisper_df: ...

    Returns:
        speakers: ...
        speaker_changes: ...
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
