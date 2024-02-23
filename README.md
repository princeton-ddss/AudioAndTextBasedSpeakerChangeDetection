## AudioAndTextBasedSpeakerChangeDetection #
AudioAndTextBasedSpeakerChangeDetection is a Python package to detect speaker change by analyzing both audio and text features.

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



## Install Package named `packageworkshop` and its dependencies
```
cd <.../AudioAndTextBasedSpeakerChangeDetection>
pip install -e .
```

## Usage
```python
import numpy as np
from packageworkshop.rescale import rescale

# rescales over 0 to 1
rescale(np.linspace(0, 100, 5))
```
