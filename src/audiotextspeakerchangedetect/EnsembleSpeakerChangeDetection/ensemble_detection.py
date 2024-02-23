import os
import pandas as pd

from .helpers import map_notsure_to_none, map_none_to_nan, use_value_major, map_string_to_bool, aggregate_two_modes

def ensemble_detection(detection_file_input_path, detection_file_input_name, ensemble_output_path,
                       ensemble_voting = ['majority', 'unanimity']):
    '''
    The main function to build an ensemble model of speaker change detection by using both audio and textual features.
    Specifically, the ensemble model is built by aggregating predictions from various detection models
    (except for the NLP rule-based model) using the input ensemble voting method.
    Then, the final prediction of the ensemble model is corrected by using results from the NLP rule-based model.

    :param detection_file_input_path: A path where the speaker change detection output in csv file is saved
    :type detection_file_input_path: str
    :param detection_file_input_name: A speaker change detection output csv file name ending with .csv
    :type detection_file_input_name: str
    :param ensemble_output_path: A path to save the ensemble detection output in csv file
    :type ensemble_output_path: str
    :param ensemble_voting: A list of voting methods to be used to build the final ensemble model
    :type ensemble_voting: list
    :return: None
    :rtype: None
    '''
    detection_csv = pd.read_csv(os.path.join(detection_file_input_path, detection_file_input_name))

    # Find all speaker change columns of all models
    speaker_change_columns = [colname for colname in detection_csv.columns if colname.startswith('speaker_change')]
    speaker_change_nonlp_columns = [colname for colname in speaker_change_columns if 'nlp' not in colname and 'ensemble' not in colname]

    if 'speaker_change_llama2' in speaker_change_columns:
        # Fill NotSure by NA: later majority voting would automatically exclude NA
        # Convert string/object type to bool type; Could not directly use astype(bool): 'False' would become True
        detection_csv['speaker_change_llama2_copy'] = detection_csv['speaker_change_llama2'].copy()
        detection_csv['speaker_change_llama2'] = detection_csv['speaker_change_llama2'].apply(
            map_notsure_to_none).apply(map_string_to_bool)

    if 'unanimity' in ensemble_voting:
        # Ensemble detection results using unanimous voting
        # Only If the ensemble model of all models agrees that the speakerchanges = FALSE, would return ensemble as false and merge sounds
        # e.g.: False or False -> False; True or False -> True
        # Reduce the false positive rate to identify speakerchanges=TURE by avoding merging sounds of different poeple together
        detection_csv['speaker_change_ensemble_unanimity'] = detection_csv[speaker_change_nonlp_columns].any(axis = 1, skipna = True)

    if 'majority' in ensemble_voting:
        # Ensemble detection results using majority voting
        # Any function takes None, but mode function takes np.NaN instead
        # Need to convert None to NaN first to use skipna = True inside mode function
        if 'speaker_change_llama2' in speaker_change_nonlp_columns:
            detection_csv['speaker_change_llama2_nan'] = detection_csv['speaker_change_llama2'].apply(map_none_to_nan)
            speaker_change_nonlp_columns_nan = speaker_change_nonlp_columns.copy()
            speaker_change_nonlp_columns_nan.remove('speaker_change_llama2')
            speaker_change_nonlp_columns_nan.append('speaker_change_llama2_nan')
            detection_csv_modes = detection_csv[speaker_change_nonlp_columns_nan].mode(axis = 1, dropna = True)
            # Consider the situation which has two modes with equal vote (True, False), under this case: take final mode as True instead of False
            if len(detection_csv_modes.columns) > 1:
                detection_csv_modes = detection_csv_modes.rename(columns={0: 'first_mode', 1: 'second_mode'})
                detection_csv_modes = detection_csv_modes.apply(lambda x: aggregate_two_modes(x['first_mode'], x['second_mode']), axis=1)
            detection_csv['speaker_change_ensemble_majority'] = detection_csv_modes
            detection_csv = detection_csv.drop(columns = ['speaker_change_llama2_nan'])
        else:
            detection_csv['speaker_change_ensemble_majority'] = detection_csv[speaker_change_nonlp_columns].mode(axis = 1, dropna = True)

    # Correct Results based on Rule-Based analysis
    # If the NLP determines the speaker changes, use NLP result to identify speaker change
    if 'speaker_change_nlp' in speaker_change_columns:
        ensemble_columns = [colname for colname in detection_csv.columns if 'speaker_change_ensemble' in colname]
        for ensemble_column in ensemble_columns:
            detection_csv[ensemble_column] = detection_csv.apply(lambda df: use_value_major(value_major= df['speaker_change_nlp'],
                                                                                                          value_minor =df[ensemble_column]), axis = 1)
            # Could not convert string/object type to bool type by using astype(bool): "False" would become True
            detection_csv[ensemble_column] = detection_csv[ensemble_column].apply(map_string_to_bool)
    # Recover speaker_change_llama2 column
    detection_csv['speaker_change_llama2'] = detection_csv['speaker_change_llama2_copy']
    detection_csv = detection_csv.drop(columns=['speaker_change_llama2_copy'])
    # Rename ensemble to include specific ways of ensembling
    detection_csv.to_csv(os.path.join(ensemble_output_path, detection_file_input_name), index = False)
    return detection_csv

