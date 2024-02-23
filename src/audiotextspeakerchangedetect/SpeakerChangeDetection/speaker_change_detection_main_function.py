from typing import Union
import pandas as pd
import torch
import os
import spacy

from .SpectralClustering.speaker_change_detection_clustering import spectralclustering_speakerchangedetection
from .PyAnnote.speaker_change_detection_pyannote import pyannote_speakerchangedetection
from .NLP.speaker_change_detection_nlp import nlp_speakerchangedetection
from .Llama2.speaker_change_detection_llama2 import llama2_speakerchangedetection

from .helpers import merge_detection_audio_results_with_transcription


def run_speaker_change_detection_models(audio_file_input_path, audio_file_input_name, min_speakers, max_speakers,
                                   transcription_input_path, transcription_input_file_name, detection_models,
                                   detection_output_path, hf_access_token,
                                   llama2_model_path, pyannote_model_path=None,
                                   device: Union[str, torch.device]=None,
                                   detection_llama2_output_path=None, temp_output_path=None):
    '''
    The main function to choose and run models to identify speaker change detections of an input
    audio file from each of models

    :param audio_file_input_path: A path which contains an input audio file
    :type audio_file_input_path: str
    :param audio_file_input_name: A audio file name containing the file type
    :type audio_file_input_name: str
    :param min_speakers: The minimal number of speakers in the input audio file
    :type: min_speakers: int
    :param max_speakers: The maximal number of speakers in the input audio file
    :type: max_speakers: int
    :param transcription_input_path: A path where a Whisper transcription output csv file is saved
    :type transcription_input_path: str
    :param transcription_input_file_name: A Whisper transcription output csv file name ending with .csv
    :type: transcription_input_file_name: str
    :param detection_models: A list of names of speaker change detection models to be run
    :type detection_models: list
    :param detection_output_path: A path to save the speaker change detection output in csv file
    :type detection_output_path: str
    :param hf_access_token: Access token to HuggingFace
    :type hf_access_token: str
    :param llama2_model_path: A path where the Llama2 model files are saved
    :type llama2_model_path: str
    :param pyannote_model_path: A path where the Pyannote model files are saved, default to None
    :type pyannote_model_path: str, optional
    :param device:Device type to run the model, defaults to None so GPU would be automatically
    used if it is available
    :type: str or torch.device, optional
    :param detection_llama2_output_path: A path where the pre-run Llama2 speaker change detection output in csv file
    is saved if exists, default to None
    :type detection_llama2_output_path: str, optional
    :param temp_output_path: A path to save the current run of Llama2 speaker change detection output
    to avoid future rerunning, default to None
    :type temp_output_path: str, optional
    :return: None
    :rtype: None
    '''

    # Check validity of inputs
    # Should only use one llama2 model for ensemble later
    llama2_models_list = [i for i in detection_models if i.startswith('llama2')] # Should be like ['llama2-7b']
    if not llama2_models_list:
        if len(llama2_models_list) > 1:
            raise Exception('Should only select one llama2 model for detection, reinput the detection models')
    
    input_csvfilename = transcription_input_file_name

    # Import whisper output
    whisper_df = pd.read_csv(os.path.join(transcription_input_path, transcription_input_file_name))
    if 'clustering' in detection_models:
        # Run detection based on the clustering of sounds
        timestamp_speaker_clustering = spectralclustering_speakerchangedetection(audio_file_input_path, audio_file_input_name,
                                                   min_speakers, max_speakers, device)

        print('finish clustering')
        speakers_clustering, speaker_change_clustering = merge_detection_audio_results_with_transcription(timestamp_speaker_clustering, whisper_df)
        whisper_df['speaker_spectralclustering'] = speakers_clustering
        whisper_df['speaker_change_spectralclustering'] = speaker_change_clustering
    if 'pyannote' in detection_models:
        timestamp_speaker_pyannote = pyannote_speakerchangedetection(audio_file_input_path, audio_file_input_name,
                                                                min_speakers, max_speakers, hf_access_token,
                                                                 device, pyannote_model_path)

        print('finish pyannote')
        speakers_pyannote, speaker_change_pyannote = merge_detection_audio_results_with_transcription(timestamp_speaker_pyannote, whisper_df)
        whisper_df['speaker_pyannote'] = speakers_pyannote
        whisper_df['speaker_change_pyannote'] = speaker_change_pyannote
    if 'nlp' in detection_models:
        nlp_model = spacy.load("en_core_web_lg")
        whisper_df['speaker_change_nlp'] = nlp_speakerchangedetection(whisper_df, nlp_model)
        print('finish nlp')

    if llama2_models_list:
        llama2_model = llama2_models_list[0]
        llama_model_size = llama2_model.split('-')[-1]
        if detection_llama2_output_path:
            df_llama2 = pd.read_csv(os.path.join(detection_llama2_output_path, input_csvfilename))
        else:
            # If no llama2 output
            # Should Run llama2 automatically
            df_llama2 = llama2_speakerchangedetection(whisper_df, llama2_model_path, llama_model_size)
            # output llama2 results for check
            if temp_output_path:
                df_llama2.to_csv(os.path.join(temp_output_path, 'llama2_results.csv'), index = False)

        # drop duplicates segment id on llama output if the output has
        df_llama2 = df_llama2.drop_duplicates(subset=['segmentid'])
        # drop text column as the audio analysis dataframe has it
        df_llama2 = df_llama2.drop(columns='text')
        # merge two dataframe based on segment id
        # Need to convert type of segmentid from object to int
        df_llama2['segmentid'] =  df_llama2['segmentid'].astype(int)
        whisper_df = pd.merge(whisper_df, df_llama2, on='segmentid', how='left')

    whisper_df.to_csv(os.path.join(detection_output_path, input_csvfilename), index=False)
    return whisper_df
