import gc
import json
import pandas as pd
import os
import torch
from typing import TypeVar

from .huggingface_offline.LLaMA_HuggingFace_Offline import setup_llama_tokenizer

from .prompts.generators import get_instructions, get_full_prompt


def llama2_speakerchangedetection(
    whisper_df: TypeVar("pandas.core.frame.DataFrame"),
    llama_model_path: str,
    llama_model_size: str,
    system_prompt: str,
    instructions_bgn: str,
    samples: str,
    device_map: str = "auto",
    torch_dtype: str = torch.float16,
):
    """
    The main function to run Llama2 to perform speaker change detection
    Please refer to the example of system_prompt, instructions_bgn, and samples from prompts/template.py
    Args:
        llama_model_path: the path to llama2 model
        llama_model_size: the size of llama2 model
        system_prompt: the system prompt
        instructions_bgn: the main question
        samples: sample question and answers
        device_map: device to run the model
        torch_dtype: the dtype of the model uses
    Returns:
        results_all_df: the transcription dataframe with speaker change detection results from llama2
    """

    # set llama_model_folder_name based on llama_model_size
    llama_model_size_local_folder_name = (
        "models--meta-llama--Llama-2-{}-chat-hf/snapshots".format(llama_model_size)
    )

    # Check if llama_model_size_local_folder exists
    # If the folder does not exist, download the model from HF
    if not os.path.exists(
        os.path.join(
            llama_model_path,
            "models--meta-llama--Llama-2-{}-chat-hf".format(llama_model_size),
        )
    ):
        raise Exception(
            "Llama model is not downloaded locally. Please download model from HF manually by following readme of repo"
        )

    # Impute Llama folder snapshot id based on llama_model_path and folder name
    # The local model folder should be like this: /Users/jf3375/Dropbox (Princeton)/models/llama/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93
    snapshot_id = [
        i
        for i in os.listdir(
            os.path.join(llama_model_path, llama_model_size_local_folder_name)
        )
        if not i.startswith(".")
    ][0]
    llama_model_size_local_folder = os.path.join(
        llama_model_size_local_folder_name, snapshot_id
    )

    # Load Llama model
    tokenizer, pipeline = setup_llama_tokenizer(
        os.path.join(llama_model_path, llama_model_size_local_folder),
        device_map,
        torch_dtype,
    )

    # Set parameters of the model to get deterministic response
    # The input includes the question itself; Need to set up max_length long enough so all the answers would return
    temperature = 0.1
    top_p = 0
    top_k = 1
    max_length = (
        4086  # The maximum number of tokens; Issues would occur if it increases
    )

    # Need to cut dataframe to ensure that the input text length does not exceed maximum tokens
    # Since token is not equal to the number of words, would divide max_length by 2 to ensure the buffer
    dataframe_segment_maxstrings = max_length - len(instructions_bgn) - len(samples)

    start_rowidx = 0
    total_num_strings = 0
    seg_num_strings = 0
    results_all_df = pd.DataFrame()

    for current_rowidx, text in enumerate(list(whisper_df["text"])):
        total_num_strings += len(text)
        seg_num_strings += len(text)
        # Consider the condition in which the segment to last row does not exceed the maximum
        if (
            seg_num_strings > dataframe_segment_maxstrings
            or current_rowidx == whisper_df.shape[0] - 1
        ):
            whisper_df_cut = whisper_df.iloc[start_rowidx : current_rowidx + 1]
            # Restart the segment
            start_rowidx = current_rowidx + 1
            seg_num_strings = 0

            # Pass the text inputs into the Llama2 model
            main_question = get_instructions(whisper_df_cut, instructions_bgn, samples)

            full_prompt = get_full_prompt(system_prompt, main_question)
            print("full prompt")
            print(full_prompt)

            # Clear Cache Memory before running Llama
            torch.cuda.empty_cache()
            gc.collect()

            # Apply Llama to get detection results
            sequences = pipeline(
                full_prompt,
                return_full_text=False,  # Not repeat the question
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=max_length,
                temperature=temperature,  # Currently does not support the set of 0
            )
            print("----------------------------------------------------------------")
            print("Answer Started")

            results_segment_df = pd.DataFrame()
            segments = []
            speakers = []
            segmentids = []

            for seq in sequences:
                txt = seq["generated_text"]
                print(f"Result: %{txt}")
                output = txt[txt.find("{") : txt.rfind("}") + 1]
                try:
                    data = json.loads(output)
                    for segment in data["conversation"]:
                        segmentids.append(segment["segment id"])
                        segments.append(segment["current segment"])
                        speakers.append(segment["speaker changes"])
                    results_segment_df["segmentid"] = segmentids
                    results_segment_df["text"] = segments
                    results_segment_df["speaker_change_llama2"] = speakers
                except:  # No Valid JSON Output
                    print("No Valid JSON Output", "identify no speaker changes", txt)
                    results_segment_df[["segmentid", "text"]] = whisper_df_cut[
                        ["segmentid", "text"]
                    ]
                    print(results_segment_df.shape[0])
                    print(whisper_df_cut.shape[0])
                    results_segment_df["speaker_change_llama2"] = [
                        ""
                    ] * whisper_df_cut.shape[0]

            results_all_df = pd.concat([results_all_df, results_segment_df])
            # Convert Empty String to NotSure to Clearly Indicate Llama2 does not identify speaker changes of those segments
            results_all_df["speaker_change_llama2"] = results_all_df[
                "speaker_change_llama2"
            ].apply(lambda x: "NotSure" if x == "" else x)
    return results_all_df
