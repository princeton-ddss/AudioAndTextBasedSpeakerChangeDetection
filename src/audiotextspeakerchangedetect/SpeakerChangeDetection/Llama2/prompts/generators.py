from typing import TypeVar

def get_full_prompt(systemprompt:str,  main_question:str, questions_answers_dict = None,
                    INST: str = '[INST]', E_INST: str = '[/INST]',
                    SYS: str = '<<SYS>>\n', E_SYS: str = '\n<</SYS>>\n\n',
                    SEN: str = '<s>', EN_SEN:str = '<\s>'):
    """
    Generate the general prompt of Llama2 following the official format
    Args:
        systemprompt: the system prompt
        main_question: the main question
        questions_answers_dict: the examples for few shot learning
    Returns:
        full_prompt: the full prompt to llama2 with few shot learning if examples are provided in the inputs
    """
    if not questions_answers_dict: # No few-shot learning
        full_prompt =   SEN + INST + \
                        SYS + systemprompt + E_SYS + \
                        main_question + E_INST + EN_SEN
    else:
        questions = list(questions_answers_dict.keys())
        full_prompt =   SEN + INST + \
                        SYS + systemprompt + E_SYS + \
                        questions[0] + E_INST + \
                        questions_answers_dict[questions[0]] + EN_SEN

        for question in questions[1:]:
            full_prompt += SEN + INST + question + E_INST + questions_answers_dict[question] + EN_SEN

        full_prompt += SEN + INST + main_question + E_INST
    return full_prompt


def get_instructions(
    whisper_df_cut:TypeVar('pandas.core.frame.DataFrame'),
    instructions_bgn:str,
    samples:str = None,
):
    """
    Generate the instructions to Llama2 to perform speaker change detection on transcriptions inputs
    Args:
        whisper_df_cut: The subset of whole whisper dataframe to not exceed the prompt length limit
        instructions_bgn: The main question
        samples: Sample question and answers
    Returns:
        instructions: the complete instructions string with the input text segments and samples
    """

    whisper_data = '{"conversation":[\n'
    segment_ids = list(whisper_df_cut["segmentid"])
    segments = list(whisper_df_cut["text"])
    previous_segment = "None"
    for idx, segment in enumerate(segments[:-1]):
        row = f'"segment id": "{segment_ids[idx]}", "previous segment": "{previous_segment}", "current segment": "{segment}", "speaker changes": ""'
        whisper_data += "{" + row + "}," + "\n"
        previous_segment = segment
    row = f'"segment id": "{segment_ids[-1]}", "previous segment": "{previous_segment}", "current segment": "{segments[-1]}", "speaker changes": ""'
    whisper_data += "{" + row + "}" + "\n"
    whisper_data += "]\n}"

    instructions = instructions_bgn + "\n" + whisper_data + "\n" + "Answer:"
    if samples:
        instructions = samples + "\n" + instructions
    return instructions
