[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10712696.svg)](https://doi.org/10.5281/zenodo.10712696)

## Audiotextspeakerchangedetect ##
**[Audiotextspeakerchangedetect](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection)** is a Python package to detect speaker change by analyzing both audio and text features.

The package develops and applies Large Language Models and the Rule-based NLP Model to detect speaker change based on text features. 

Currently, the package provides the main function so users could directly pass transcriptions to apply Llama2-70b to detect speaker change. The prompt of speaker change detection 
is developed meticulously to ensure that Llama2 could understand its role of detecting speaker change, perform the speaker change detection for almost every segment, and return the answer in a standardized JSON format. 
Specifically, two texts of the current segment and the next segment would be shown to ask llama2 if the speaker changes across these two segments by understanding the interrelationships 
between these two texts via their semantic meaning. The codes are developed to parse input csv files to prompts and parse the returned answers into csv files
while considering possible missing values and mismatches. 

In addition to Llama2, the Rule-based NLP model is also developed to detect speaker change by analyzing the text using human comprehension. Well-defined patterns exist in the text segments 
so humans could use them to identify that the speaker indeed changes across these text segments with nearly complete certainty. 
By using Spacy NLP model, human comprehension could be written as rules in programming language. 
These rules are used to determine if these well-defined patterns exist in text segments to identify if speaker changes across these segments. 
These human-specified rules are developed by analyzing OpenAI Whisper transcription text segments. Specifically, the rules are below.
 * If the segment starts with the lowercase character, the segment continues the previous sentence. The speaker does not change in this segment.
 * If the sentence ends with ?, and its following sentence ends with . The speaker changes in the next segment.
 * If there is the conjunction word in the beginning of segment. The speaker does not change in this segment.

Besides text features, audio features are used to detect speaker change via the widely used clustering method, PyAnnotate and Spectral Clustering.

In the end, the Ensemble Audio-and-text-based Speaker Change Detection Model is built by aggregating predictions across all the speaker change detection models. The voting methods would be used to ensemble the speaker change detection models above except for Rule-based NLP model.
The predictions would be finally corrected by Rule-based NLP model as final predictions.

## Create New Python Environment to Avoid Packages Versions Conflict If Needed
```
python -m venv <envname>
source <envname>/bin/activate
```

## Install **Audiotextspeakerchangedetect** using Pypi
```
pip install audiotextspeakerchangedetect
```

## Install **Audiotextspeakerchangedetect** using Github
```
git lfs install
git clone https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection
cd <.../AudioAndTextBasedSpeakerChangeDetection>
pip install .
```

## Download Models Offline to Run Them without Internet Connection
### Download Spacy NLP Model by Running Commands below in Terminal
```
python -m spacy download en_core_web_lg
```

### Download Llama2 Model by Running Codes below in Python
<hf_access_token> is the access token to Hugging Face.
Please create a [Hugging Face account](https://huggingface.co/) if it does not exist.  
The new access token could be created by following the [instructions](https://huggingface.co/docs/hub/en/security-tokens).

<download_model_path> is the local path where the downloaded Llama2 model would be saved.
```
from huggingface_hub import snapshot_download, login

login(token=<hf_access_token>)
snapshot_download(repo_id ='meta-llama/Llama-2-70b-chat-hf',  cache_dir= <download_model_path>)
```

### Download PyAnnotate Models using Git Large File Storage (LFS)

PyAnnotate models are already in the **models** folder of the current repo. 

Please download the models using ```git lfs pull```.
To use the PyAnnotate models, please replace <local_path> with the local parent folder of the downloaded AudioAndTextBasedSpeakerChangeDetection repo in **models/pyannote3.1/Diarization/config.yaml** and
**models/pyannote3.1/Segmentation/config.yaml**.


## Usage
The audio-and-text-based ensemble speaker change detection model could be applied to get speaker change detection results by running only one function.
The function is **run_ensemble_audio_text_based_speaker_change_detection_model** in src/audiotextspeakerchangedetect/main.py.

Please view the sample codes to run the function in sample_run.py and sample_run_existingllama2output.py in the src/audiotextspeakerchangedetect.

Specifically, detection_models input could be set as a list or sublist of 
['pyannote', 'clustering', 'nlp', 'llama2-70b']. Device input could be set as None or 'gpu' or 'cpu'. If device is set as None,
gpu would be used if it is available. Running llama2-70b requires at least 2 gpus and 250GB memory. If the computing resources is not available
for running llama2-70b, please exclude llama2-70b from detection_models input or replace llama2-70b with llama2-7b.

Please view the detailed function description and its inputs descriptions inside the Python file **src/audiotextspeakerchangedetect/main.py**. 
```
from audiotextspeakerchangedetect.main import run_ensemble_audio_text_based_speaker_change_detection_model

run_ensemble_audio_text_based_speaker_change_detection_model(detection_models, min_speakers, max_speakers,
                                                           audio_file_input_path, audio_file_input_name,
                                                           transcription_input_path, transcription_file_input_name,
                                                           detection_output_path,  hf_access_token,
                                                           llama2_model_path, pyannote_model_path, device,
                                                           detection_llama2_output_path, temp_output_path, ensemble_voting)
```


## Evaluation
[VoxConverse Dataset v0.3](https://github.com/joonson/voxconverse?tab=readme-ov-file)

VoxConverse is an only audio-visual diarization dataset consisting of over 50 hours of multispeaker clips of human speech, extracted from YouTube videos, usually in a political debate or news segment context to ensure multi-speaker dialogue.
The audio files in the dataset have lots of variations of the proportion of speaker changes, which indicates the effectiveness of the dataset as the evaluation dataset to evaluate the models robustness.

Average Coverage, Purity, Precision, and Recall

|           | PyAnnote | Llama2 | Unanimity | Majority | 
|-----------|----------|--------|-----------|----------|
| Coverage  | 86%      | 45%    | 59%       | 84%      | 
| Purity    | 83%      | 89%    | 87%       | 70%      | 
| Precision | 23%      | 14%    | 24%       | 32%      | 
| Recall    | 19%      | 32%    | 41%       | 19%      | 


[AMI Headset Mix](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)

The AMI Meeting Corpus is a multi-modal data set consisting of 100 hours of meeting recordings. Around two-thirds of the data has been elicited using a scenario in which the participants play different roles in a design team, taking a design project from kick-off to completion over the course of a day. The rest consists of naturally occurring meetings in a range of domains.
Different from VoxConverse Dataset, AMI dataset is not that diverse as it only consists of meeting recording. The median and average proportion of speaker change is both around 78%, and the minimal speaker change is above 59%. Thus, the evaluation analysis based on AMI is more applicable to measure the models performance under regular conversational setting.

Average Coverage, Purity, Precision, and Recall

|           | PyAnnote | Llama2 | Unanimity | Majority | 
|-----------|----------|--------|-----------|----------|
| Coverage  | 89%      | 75%    | 80%       | 92%      | 
| Purity    | 60%      | 65%    | 64%       | 46%      | 
| Precision | 44%      | 32%    | 40%       | 46%      | 
| Recall    | 18%      | 18%    | 25%       | 11%      | 


For the detailed evaluation analysis, please refer to evaluation_analysis.pdf in the main repo folder.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
