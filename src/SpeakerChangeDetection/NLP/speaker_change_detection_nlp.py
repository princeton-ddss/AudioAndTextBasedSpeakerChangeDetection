def nlp_speakerchangedetection(whisper_df, nlp_model):
    speaker_changes_list = ['NotSure'] * whisper_df.shape[0]
    texts_list = list(whisper_df['text'])

    prev_end_token = None

    for idx, text in enumerate(texts_list):
        text_tokens = nlp_model(text)

        # If this segment ends with ., and its previous segment ends with ?: speaker changes in this segment
        if idx != 0:
            if text_tokens[-1].pos_ == 'PUNCT' and prev_end_token.pos_ == 'PUNCT':
                if text_tokens[-1].text == '.' and prev_end_token.text == '?':
                    speaker_changes_list[idx] = True

        # Put the most determined rules in the end
        # If a conjunction word is in the beginning of the sentence, speaker does not change in this segment
        if text_tokens[0].pos_ == 'CCONJ':
            speaker_changes_list[idx] = False

        # If the segment starts with the lowercase character, the segment continues the previous sentence.
        # Speaker does not change in this segment
        if text_tokens[0].is_alpha:
            if text_tokens[0].is_lower:
                speaker_changes_list[idx] = False

        # Set previous tokens as current at the end of the loop
        prev_end_token = text_tokens[-1]
    return speaker_changes_list


