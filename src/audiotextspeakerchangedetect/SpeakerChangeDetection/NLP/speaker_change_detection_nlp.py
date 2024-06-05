from typing import TypeVar
from spacy.language import Language


def nlp_speakerchangedetection(
    whisper_df: TypeVar('pandas.core.frame.DataFrame'),
    nlp_model: Language
):
    """
    Perform speaker change detection using Rule-based NLP
    Args:
        whisper_df: the pandas dataframe of whisper transcriptions
        nlp_model: the loaded spacy nlp model
    """

    speaker_changes = ["NotSure"] * whisper_df.shape[0]
    text_segments = list(whisper_df["text"])
    prev_end_token = None

    for idx, text in enumerate(text_segments):
        text_tokens = nlp_model(text)
        # Check the rules in increasing orders of certainty: The higher certainty rule results would replace the
        # lower certainty rule results

        # Segment ends with "." and previous segment ends with "?" => speaker changes.
        if idx != 0:
            if text_tokens[-1].pos_ == "PUNCT" and prev_end_token.pos_ == "PUNCT":
                if text_tokens[-1].text == "." and prev_end_token.text == "?":
                    speaker_changes[idx] = True
        
        # Conjunction word is in the beginning of the sentence => speaker does NOT change.
        if text_tokens[0].pos_ == "CCONJ":
            speaker_changes[idx] = False
        
        # Segment starts with the lowercase character => the segment continues the previous sentence => speaker does NOT change.
        if text_tokens[0].is_alpha:
            if text_tokens[0].is_lower:
                speaker_changes[idx] = False

        # Set the current token as the previous token at the end of the loop
        prev_end_token = text_tokens[-1]
    return speaker_changes
