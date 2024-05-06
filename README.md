[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10712695.svg)](https://doi.org/10.5281/zenodo.10712695)

## AudioTextSpeakerChangeDetect

**[AudioTextSpeakerChangeDetect](https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection)** is a Python package to detect speaker changes by analyzing both audio and textual features.

The package applies Large Language Models and rule-based NLP algorithms to detect speaker changes based on textual features. Users can directly pass transcriptions to apply Llama2-70b to detect speaker changes using a carefully engineered that instructs Llama2 to perform the speaker change detection for every text segment and return its answer in a standardized JSON format.

In addition to Llama2, the package provides a rule-based NLP correction model that detects speaker changes based on the presence or absence of the following natural language patterns:

- If a segment starts with a lowercase character, the segment continues the previous sentence. The speaker does not change in this segment.
- If a sentence ends with `?` and its following sentence ends with `.`, then the following sentence is a new speaker.
- If there a conjunction word (`and`, `or`, `also`, etc.) at the beginning of a segment, then there is no speaker change.

Text-based detections can be combined with audio-based detections to improve overall speaker change detection. `AudioTextSpeakerChangeDetect` uses clustering methods provided by `PyAnnote` and `spectralcluster` for audio-based speaker change detection, and ensemble (audio plus text) detection model are built by aggregating predictions across constituent detection models. The package supports the following ensemble methods out-of-the-box:

- **Majority vote**. A speaker change is predicted if the majoirty of models indicate a change.
- **Single vote**. A speaker change is predicted if _any_ model detects a change.

The rule-based NLP correction model is applied to ensemble predctions to obtain final predictions. Specifically, the full model predicts a speaker change if and only if the ensemble model and the rule-based NLP model predict a change.

## Installation

### 1. Create a new Python environment (recommended)

```
python -m venv <envname>
source <envname>/bin/activate
```

### 2. Install the package

The package **Audiotextspeakerchangedetect** can be installed via Pypi or Github.

#### Pypi (stable version)

```
pip install audiotextspeakerchangedetect
```

### Github (latest version)

```
git lfs install
git clone https://github.com/princeton-ddss/AudioAndTextBasedSpeakerChangeDetection
cd AudioAndTextBasedSpeakerChangeDetection
pip install .
```

## 3. Download models (optional)

In case your work environment does not have internet access (e.g., a Slurm job on a firewalled compute node), pre-download core model files for offline access.

### Spacy NLP

```
python -m spacy download en_core_web_lg
```

### Llama2

An access token is required to download Llama2 from Hugging Face. Create a [Hugging Face account](https://huggingface.co/) (if you don't already have one) and follow the [instructions](https://huggingface.co/docs/hub/en/security-tokens) to create a secure token. Then, run the following Python script:

```
from huggingface_hub import snapshot_download, login

login(token=<hf_access_token>)
snapshot_download(repo_id ='meta-llama/Llama-2-70b-chat-hf',  cache_dir= <download_model_path>)
```

where `<hf_access_token>` is the access token to Hugging Face and `<download_model_path>` is the local path where the downloaded Llama2 model would be saved.

### PyAnnote

PyAnnotate models can be downloaded from the `pyannote3.1` folder in this [Dropbox Link](https://www.dropbox.com/scl/fo/tp2uryaq81sze2l0yuxb9/ACgXWOr7Be1ZZovz7xNSuTs?rlkey=9c2z50pjbjhoo3vz4dbxlmlcf&st=fukejg4l&dl=0). To use the these models, replace `<local_path>` with the local parent folder of the downloaded `pyannote3.1` folder in `pyannote3.1/Diarization/config.yaml` and `pyannote3.1/Segmentation/config.yaml`.

## Usage

Audio-and-text-based ensemble speaker change predictions are obtained by running the **run_ensemble_audio_text_based_speaker_change_detection_model** function:

```python
from audiotextspeakerchangedetect.main import run_ensemble_audio_text_based_speaker_change_detection_model

detection_models = ['pyannote', 'clustering', 'nlp', 'llama2-70b']
min_speakers = 2
max_speakers = 10
audio_file_input_path = '~/transcripts/in'
audio_file_input_name =  'sample.wav'
transcription_input_path = '~/input'
transcription_file_input_name = 'sample.csv'
detection_output_path =  '~/transcripts/out'
hf_access_token = 'secret'
llama2_model_path = '~/models/llama'
pyannote_model_path = "~/models/pyannote3.1/Diarization"
device = None  # use gpu if cuda is available, otherwise use cpu
detection_llama2_output_path =  None # no existing llama2 output
temp_output_path = '/tmp'
ensemble_voting = ['majority', 'unanimity']

run_ensemble_audio_text_based_speaker_change_detection_model(
    detection_models,
    min_speakers,
    max_speakers,
    audio_file_input_path,
    audio_file_input_name,
    transcription_input_path,
    transcription_file_input_name,
    detection_output_path,
    hf_access_token,
    llama2_model_path,
    pyannote_model_path,
    device,
    detection_llama2_output_path,
    temp_output_path,
    ensemble_voting
)
```
Please note that running `llama2-70b` requires at least 250GB GPU memory. If sufficient computing resources are not available to run llama2-70b, exclude llama2-70b from `detection_models`.

## Evaluation

[VoxConverse Dataset v0.3](https://github.com/joonson/voxconverse?tab=readme-ov-file)

VoxConverse is an audio-visual diarization dataset consisting of over 50 hours of multi-speaker clips of human speech extracted from YouTube videos. The clips generally feature political debate or news segments that ensure multi-speaker dialogue.
The audio files include a wide range of speaker change frequencies, which makes this a challenging evaluation dataset.

|           | PyAnnote | Llama2 | Unanimity | Majority |
| --------- | -------- | ------ | --------- | -------- |
| Coverage  | 86%      | 45%    | 59%       | 84%      |
| Purity    | 83%      | 89%    | 87%       | 70%      |
| Precision | 23%      | 14%    | 24%       | 32%      |
| Recall    | 19%      | 32%    | 41%       | 19%      |

**Table** Average detection performance for the `VoxConverse` dataset.

[AMI Headset Mix](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)

The AMI Meeting Corpus is a multi-modal data set consisting of 100 hours of meeting recordings. Around two-thirds of the data is elicited from a scenario in which participants play different roles in a design team. The rest consists of naturally occurring meetings in a range of domains.

Relative to the VoxConverse dataset, the AMI is not that diverse as it only contains meeting recordings. The median and average proportion of speaker change are both around 78% and the minimal proportion is above 59%. Thus, the evaluation analysis based on AMI is more representative of a typical conversational setting.

|           | PyAnnote | Llama2 | Unanimity | Majority |
| --------- | -------- | ------ | --------- | -------- |
| Coverage  | 89%      | 75%    | 80%       | 92%      |
| Purity    | 60%      | 65%    | 64%       | 46%      |
| Precision | 44%      | 32%    | 40%       | 46%      |
| Recall    | 18%      | 18%    | 25%       | 11%      |

**Table** Average detection performance for the `AMI Headset Mix` dataset.

For the detailed evaluation analysis, please refer to `evaluation_analysis.pdf` in the main repo folder.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
