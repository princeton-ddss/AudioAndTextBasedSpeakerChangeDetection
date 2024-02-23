def get_main_question(whisper_df_cut, main_question_bgn, examples = None):
    whisper_data = '{"conversation":[\n'
    segmentids = list(whisper_df_cut['segmentid'])
    segments = list(whisper_df_cut['text'])

    previous_segment = 'None'
    for idx, segment in enumerate(segments[:-1]):
        whole_row = '"segment id": "{}", "previous segment": "{}", "current segment": "{}", "speaker changes": ""'.format(segmentids[idx], previous_segment, segment)
        whisper_data += '{' + whole_row + '},' + '\n'
        previous_segment = segment

    # Last row, no , after json format
    whole_row = '"segment id": "{}", "previous segment": "{}", "current segment": "{}", "speaker changes": ""'.format(segmentids[-1], previous_segment, segments[-1])
    whisper_data += '{' + whole_row + '}' + '\n'

    # Complete JSON format
    whisper_data+= ']\n}'

    main_question = main_question_bgn + '\n' +  whisper_data + '\n' + 'Answer:'

    if examples:
        main_question = examples + '\n' + main_question
    return main_question

