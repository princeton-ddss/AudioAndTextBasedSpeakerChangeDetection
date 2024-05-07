INST = ("[INST]",)
E_INST = "[/INST]"
SYS = "<<SYS>>\n"
E_SYS = "\n<</SYS>>\n\n"
SEN = "<s>"
EN_SEN = "<\s>"


def get_full_prompt(
    system_prompt: str,
    instructions: str,
    questions_answers_dict: dict = None,
):
    """..."""

    if not questions_answers_dict:  # No few-shot learning
        full_prompt = (
            SEN + INST + SYS + system_prompt + E_SYS + instructions + E_INST + EN_SEN
        )
    else:
        questions = list(questions_answers_dict.keys())
        full_prompt = (
            SEN
            + INST
            + SYS
            + system_prompt
            + E_SYS
            + questions[0]
            + E_INST
            + questions_answers_dict[questions[0]]
            + EN_SEN
        )
        for question in questions[1:]:
            full_prompt += (
                SEN
                + INST
                + question
                + E_INST
                + questions_answers_dict[question]
                + EN_SEN
            )
        full_prompt += SEN + INST + instructions + E_INST

    return full_prompt


def get_instructions(
    # TODO: add type hints
    whisper_df_cut,
    instructions_bgn,
    examples=None,
):
    """..."""

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
    if examples:
        instructions = examples + "\n" + instructions
    return instructions
