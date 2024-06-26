'''
Sample File to Run Speaker Change Detection Models With Existing Llama2 output
'''

from audiotextspeakerchangedetect.main import run_ensemble_audio_text_based_speaker_change_detection_model

# Run Detection Model with existing llama2 output
detection_models =  ['pyannote', 'clustering', 'nlp', 'llama2-70b']
min_speakers = 2
max_speakers = 10
audio_file_input_path = '/scratch/gpfs/jf3375/modern_family/audio/sample_data'
audio_file_input_name =  'sample_data.WAV'
transcription_input_path = '/scratch/gpfs/jf3375/modern_family/output/Whispertimestamped'
transcription_file_input_name = audio_file_input_name.split('.')[0] + '.csv'
detection_output_path =  '/scratch/gpfs/jf3375/modern_family/output/detection'
hf_access_token = '<hf_access_token>'
llama2_model_path = None
pyannote_model_path = "/scratch/gpfs/jf3375/models/pyannote3.1/Diarization"
device = None  # if set device = None, by default would use gpu if cuda is available, otherwise use gpu
detection_llama2_output_path =  '/scratch/gpfs/jf3375/modern_family/output/detection/Llama2/70b' # Existing llama2 output
temp_output_path = '/scratch/gpfs/jf3375/modern_family/temp'
ensemble_voting = ['majority', 'singularity']

run_ensemble_audio_text_based_speaker_change_detection_model(detection_models, min_speakers, max_speakers,
                                                           audio_file_input_path, audio_file_input_name,
                                                           transcription_input_path, transcription_file_input_name,
                                                           detection_output_path,  hf_access_token,
                                                           llama2_model_path, pyannote_model_path, device,
                                                           detection_llama2_output_path, temp_output_path, ensemble_voting)


