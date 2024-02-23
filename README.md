## Audiotextspeakerchangedetect ##
**Audiotextspeakerchangedetect** is a Python package to detect speaker change by analyzing both audio and text features.

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

In the end, the ensemble audio-text based speaker change detection model is built by aggregating predictions across all the speaker change detection models via voting method.

The evaluation module is also developed inside package to evaluate the speaker change detection models performance by using SegmentationPrecision,
SegmentationRecall, SegmentationCoverage, and SegmentationPurity.

## Create New Python Environment to Avoid Packages Versions Conflict If Needed
```
python -m venv <envname>
source <envname>/bin/activate
```

## Install **Audiotextspeakerchangedetect** 
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
Please create a Hugging Face account (https://huggingface.co/) if it does not exist.  
The new access token could be created by following the [instructions](https://huggingface.co/docs/hub/en/security-tokens).

<download_model_path> is the local path where the downloaded Llama2 model would be saved.
```
from huggingface_hub import snapshot_download, login

login(token=<hf_access_token>)
snapshot_download(repo_id ='meta-llama/Llama-2-70b-chat-hf',  cache_dir= <download_model_path>)
```

### Download PyAnnotate Models using Git Large File Storage (LFS)

PyAnnotate models are already in the **models** folder of the current repo. 

To use the PyAnnotate models, please replace <local_path> with the local parent folder of the downloaded AudioAndTextBasedSpeakerChangeDetection repo in **models/pyannote3.1/Diarization/config.yaml** and
**models/pyannote3.1/Segmentation/config.yaml**.


## Usage
The audio-and-text-based ensemble speaker change detection model could be applied to get speaker change detection results by running only one function.
The function is **run_ensemble_audio_text_based_speaker_change_detection_model** in src/audiotextspeakerchangedetect/main.py.
Please view the function description inside the Python file.
Please view the sample codes to run the function in sample_run.py and sample_run_existingllama2output.py in the src/audiotextspeakerchangedetect.
```
from audiotextspeakerchangedetect.main import run_ensemble_audio_text_based_speaker_change_detection_model

run_ensemble_audio_text_based_speaker_change_detection_model(detection_models, min_speakers, max_speakers,
                                                           audio_file_input_path, audio_file_input_name,
                                                           transcription_input_path, transcription_file_input_name,
                                                           detection_output_path,  hf_access_token,
                                                           llama2_model_path, pyannote_model_path, device,
                                                           detection_llama2_output_path, temp_output_path, ensemble_voting)
```


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
