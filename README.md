# Multilingual Automatic Speech Recognition Evaluation

A multilingual speech-recognition project comparing **DeepSpeech** and **Vosk** across English, Italian and Spanish audio.

The project evaluates automatic speech recognition output against fixed reference transcriptions using **Word Error Rate (WER)**. It also includes speech preprocessing for the Vosk pipeline and two additional user-recorded English sentences for extended testing.

## Project scope

The project contains two speech-recognition workflows:

- **DeepSpeech** evaluation using English, Spanish and Italian acoustic/language models;
- **Vosk offline ASR** evaluation using separate language-specific models for English, Spanish and Italian.

Both workflows compare generated transcriptions with reference text and export quantitative results for analysis.

## Languages

The supplied audio set contains:

- English
- Italian
- Spanish

The main supplied set contains five standard recordings per language. Two additional English recordings are also included for user-recorded sentence testing.

## Audio dataset

The repository includes the original WAV files used in the project.

### English

```text
checkin.wav
parents.wav
suitcase.wav
what_time.wav
where.wav
your_sentence1.wav
your_sentence2.wav
```

### Italian

```text
checkin_it.wav
parents_it.wav
suitcase_it.wav
what_time_it.wav
where_it.wav
```

### Spanish

```text
checkin_es.wav
parents_es.wav
suitcase_es.wav
what_time_es.wav
where_es.wav
```

The reference sentences represent short airport/travel queries such as check-in locations, lost luggage, flight times and restaurant/shop locations.

## DeepSpeech workflow

The original DeepSpeech notebook is:

```text
deepspeech/Exercise4_DeepSpeech.ipynb
```

The associated Python implementation is:

```text
deepspeech/Exercise4.py
```

The notebook documents the use of:

- DeepSpeech 0.9.3 for English;
- Scribosermo community DeepSpeech models for Spanish and Italian;
- model/scorer pairs for language-specific decoding;
- word-level transcript normalization;
- custom WER calculation using dynamic programming / Levenshtein edit distance;
- CSV-based prediction and evaluation outputs.

### DeepSpeech execution environment

The notebook records that DeepSpeech 0.9.x was not run natively on the Apple Silicon/Python environment used for the coursework. The DeepSpeech inference stage was therefore executed through **Docker**, with predictions stored in CSV files for subsequent WER analysis.

The original model files are not included in this repository.

Expected model/scorer names documented by the project include:

```text
English:
deepspeech-0.9.3-models.pbmm
deepspeech-0.9.3-models.scorer

Spanish / Italian:
model.pbmm
kenlm.scorer
```

## Vosk workflow

The original Vosk notebook is:

```text
vosk/Exercise4-vosk.ipynb
```

It implements offline speech recognition using separate language-specific Vosk models.

### Audio preprocessing

Before recognition, the Vosk workflow standardizes audio using:

1. WAV loading;
2. conversion to floating-point samples;
3. mono conversion where required;
4. resampling to **16 kHz**;
5. RMS loudness normalization;
6. optional Butterworth speech-band filtering;
7. post-filter normalization;
8. conversion back to `int16` PCM for recognition.

The optional speech-focused bandpass filter uses approximately:

```text
300 Hz – 3400 Hz
```

### Language-specific recognition

The notebook maps each input file to the correct Vosk language model and caches loaded models so they do not need to be reloaded for every recording.

The original notebook expects model directories such as:

```text
EN/model-en
ES/model-es
IT/model-it
```

The large external Vosk model directories are not included in this repository.

## Word Error Rate evaluation

Both workflows evaluate recognition output using **Word Error Rate**.

WER measures the number of word substitutions, deletions and insertions required to transform a hypothesis transcript into its reference transcript.

The Vosk notebook also records the individual:

```text
S = substitutions
D = deletions
I = insertions
N = number of reference words
```

Text is normalized before comparison to reduce mismatches caused only by casing, punctuation, apostrophe style and whitespace.

## Recorded Vosk results

The original Vosk notebook evaluates the five standard recordings for each language and reports the following mean WER values:

| Language | Files | Mean WER |
|---|---:|---:|
| English | 5 | 15.33% |
| Italian | 5 | 7.86% |
| Spanish | 5 | 26.91% |

Two additional English recordings are then evaluated and added to the final `vosk_results.csv` output.

The final stored Vosk CSV contains **17 evaluated recordings**.

## Recorded DeepSpeech results

The repository preserves the original DeepSpeech result files:

```text
deepspeech_results_per_file.csv
deepspeech_results_wer_summary.csv
ex4_predictions.csv
ex4_predictions_with_wer.csv
```

The stored DeepSpeech summary reports mean WER values of approximately:

| Language | Mean WER |
|---|---:|
| English | 21.22% |
| Italian | 45.43% |
| Spanish | 41.17% |

The DeepSpeech result files were generated at different stages of the coursework and do not all contain exactly the same English file set. They are therefore retained exactly as produced rather than being rewritten or reconciled.

## Custom English recordings

The Vosk workflow also evaluates:

```text
your_sentence1.wav
your_sentence2.wav
```

against user-defined reference transcripts.

This extends the supplied dataset with independently recorded speech and allows the same WER pipeline to be applied to new input.

## Repository structure

```text
multilingual-asr-evaluation/
├── README.md
├── requirements.txt
├── .gitignore
├── ORIGINAL_FILES_VERIFIED.txt
├── Ex4_audio_files/
│   ├── EN/
│   ├── ES/
│   └── IT/
├── deepspeech/
│   ├── Exercise4.py
│   ├── Exercise4_DeepSpeech.ipynb
│   ├── deepspeech_results_per_file.csv
│   ├── deepspeech_results_wer_summary.csv
│   ├── ex4_predictions.csv
│   └── ex4_predictions_with_wer.csv
└── vosk/
    ├── Exercise4-vosk.ipynb
    └── vosk_results.csv
```

## Dependencies

The notebook workflows use:

- Python
- NumPy
- pandas
- SciPy
- Vosk
- Jupyter Notebook
- DeepSpeech 0.9.x
- Docker for the DeepSpeech environment used in the project

A lightweight Python dependency list is included in `requirements.txt`.

DeepSpeech itself is intentionally not pinned in that file because the original notebook documents a Docker-based DeepSpeech 0.9.x environment rather than a native installation on the development machine.

## Running the Vosk notebook

Install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download appropriate English, Spanish and Italian Vosk models and place them in the locations expected by the notebook.

Then start Jupyter:

```bash
jupyter notebook
```

Open:

```text
vosk/Exercise4-vosk.ipynb
```

### Local path note

The original Vosk notebook contains the local `AUDIO_DIR` path used when the project was executed. To run it on another machine, that path must point to the local `Ex4_audio_files` directory.

The notebook itself is preserved unchanged in this repository.

## Running the DeepSpeech analysis

The DeepSpeech notebook documents its original Docker-based inference workflow. Existing generated prediction CSV files are included, so the WER analysis and recorded outputs can be inspected without regenerating all model predictions.

## Original-file preservation

The original source notebooks, Python code, WAV recordings and CSV result files have been copied without rewriting their contents.

Only repository documentation/support files were added. Operating-system metadata and Jupyter autosave checkpoint folders are excluded.

## Techniques demonstrated

**Automatic speech recognition:** DeepSpeech, Vosk, multilingual decoding  
**Audio preprocessing:** mono conversion, resampling, RMS normalization, bandpass filtering, PCM conversion  
**Signal processing:** SciPy filtering and polyphase resampling  
**Evaluation:** WER, substitutions, deletions, insertions, per-language analysis  
**Data handling:** pandas result tables and CSV export  
**Environment management:** Docker-based DeepSpeech execution and offline Vosk models

## Author

**Zimmel Javed Virk**  
BSc Computer Science — University of London
