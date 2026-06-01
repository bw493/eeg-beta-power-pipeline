# EEG Beta Power Analysis Pipeline: Prerequisites and Setup Guide

> **Document type**: SOP Prerequisite / Supplement for `eeg_pipeline.py`
> **Project**: EEG Beta Power Analysis Pipeline (CruX Neurotechnology Club, UCLA)
> **Last updated**: June 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Software Prerequisites](#3-software-prerequisites)
4. [Hardware Prerequisites](#4-hardware-prerequisites)
5. [Data Format Specification](#5-data-format-specification)
6. [Step 1: Raw EEG to Trial Data (`eeg_to_trials.py`)](#6-step-1-raw-eeg-to-trial-data-eeg_to_trialspy)
7. [Step 2: Analysis Pipeline (`eeg_pipeline.py`)](#7-step-2-analysis-pipeline-eeg_pipelinepy)
8. [Output Interpretation](#8-output-interpretation)
9. [Common Errors and Remediation](#9-common-errors-and-remediation)
10. [Glossary](#10-glossary)

---

## 1. Project Overview

This pipeline processes raw OpenBCI EEG recordings and extracts per-trial beta power features in relation to behavioral performance on a letter-identification task. It supports two participant groups: **control** (baseline only) and **neurofeedback** (baseline followed by feedback intervention). The pipeline produces correlation statistics, correct-versus-missed beta power comparisons, and block-level power trends.

The full workflow proceeds in two sequential stages:

```
Raw OpenBCI .txt file
        │
        ▼
  eeg_to_trials.py   ←── assigns trial epochs + extracts spectral features
        │
        ▼
  trials_output.csv  ←── one row per trial, with beta/alpha/theta power
        │
        ▼
  eeg_pipeline.py    ←── statistical analysis + visualization
        │
        ▼
  eeg_pipeline_results.png
```

**`eeg_pipeline.py` cannot be run correctly without first producing a well-formed `trials_output.csv` via `eeg_to_trials.py`.** This document specifies all prerequisites for both scripts.

> **Note on interface approach**: A prior CruX BCI project (ssvep-bci-openbci) exposes its pipeline through a web-based interface. This project does not follow that pattern. All analysis is executed directly through eeg_pipeline.py from the terminal. No browser or web server is required.


---

## 2. Repository Structure

```
EEG Beta Power Analysis Pipeline/
├── eeg_pipeline.py                          # Stage 2: statistical analysis and visualization
├── eeg_to_trials.py                         # Stage 1: raw EEG to per-trial feature extraction
├── eeg_trial_data.csv                       # Example/legacy trial data (23-trial dataset)
├── OpenBCI-RAW-2026-02-23_19-06-24.txt      # Raw OpenBCI recording (primary input)
├── BrainFlow-RAW_2026-02-23_19-05-38_0.csv  # Alternative BrainFlow-format recording
├── neuroflow-eeg-analytics.zip              # Archived React + Vite frontend (not tracked as submodule)
├── .gitignore                               # Excludes generated files and system artifacts
└── README.docx
```

> **Note on data files**: `eeg_trial_data.csv` is an earlier 23-trial dataset (likely a manually synchronized version). `trials_output.csv` is the programmatically generated 80-trial dataset produced by `eeg_to_trials.py`. The pipeline can accept either, but only `trials_output.csv` reflects the full recording session.

---

## 3. Software Prerequisites

### 3.1 Python Version

| Requirement | Specification |
|-------------|---------------|
| Python | 3.9 or higher (tested on 3.14 via `/opt/homebrew/bin/python3`) |
| pip | Latest stable version |

To verify your Python installation:

```bash
python3 --version
which python3
```

### 3.2 Required Python Packages

Install all dependencies before running either script:

```bash
pip install numpy scipy pandas matplotlib mne
```

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and spectral computation |
| `scipy` | Signal filtering, Welch PSD, statistical tests |
| `pandas` | CSV I/O and trial data manipulation |
| `matplotlib` | Result visualization and figure export |
| `mne` | Epoch construction and artifact rejection (used by `eeg_pipeline.py`) |

> **macOS note**: If you are using a Homebrew-managed Python (as in this project), prefer `pip3 install --break-system-packages <package>` or use a virtual environment to avoid system-level conflicts.

### 3.3 Optional: Virtual Environment Setup (Recommended)

```bash
cd "EEG Beta Power Analysis Pipeline"
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib mne
```

Activate this environment before every session:

```bash
source .venv/bin/activate
```

---

## 4. Hardware Prerequisites

### 4.1 OpenBCI Cyton Board

This pipeline is designed for recordings made with the **OpenBCI Cyton** board (8-channel, 250 Hz sample rate). The raw input file must conform to the OpenBCI text format (see Section 5).

| Parameter | Value |
|-----------|-------|
| Channels | 8 (EEG) |
| Sample rate | 250 Hz |
| Recording format | OpenBCI RAW `.txt` |
| Expected recording duration | ~115 seconds (for 80 trials) |

### 4.2 BrainFlow Format Support

`eeg_to_trials.py` also supports BrainFlow-format recordings via the `--format brainflow` flag. The BrainFlow file in this repository (`BrainFlow-RAW_2026-02-23_19-05-38_0...`) can be used as an alternative input.

---

## 5. Data Format Specification

### 5.1 Raw OpenBCI File

The input to `eeg_to_trials.py` is a standard OpenBCI RAW text file. The file must begin with the OpenBCI header block (lines prefixed with `%`) followed by comma-separated sample rows.

Expected columns (0-indexed):

| Index | Content |
|-------|---------|
| 0 | Sample index |
| 1–8 | EEG channel voltages (µV) |
| 9–11 | Accelerometer axes |
| 12+ | Timestamp and aux data |

### 5.2 Generated Trial CSV (`trials_output.csv`)

`eeg_to_trials.py` produces a CSV with the following schema, which `eeg_pipeline.py` reads as input:

| Column | Type | Description |
|--------|------|-------------|
| `trial_id` | int | Sequential trial index (1–N) |
| `block` | int | Block number (1–4 for a 4-block session) |
| `stimulus_letter` | str | Letter shown to participant |
| `key_pressed` | str | Key pressed by participant |
| `correct` | int | 1 if correct, 0 if incorrect |
| `reaction_time_ms` | float | Reaction time in milliseconds |
| `beta_power_db` | float | Beta band power (13–30 Hz) in dB relative to broadband |
| `alpha_power_db` | float | Alpha band power (8–13 Hz) in dB |
| `theta_power_db` | float | Theta band power (4–8 Hz) in dB |
| `trial_onset_s` | float | Trial onset time in seconds from recording start |
| `participant` | str | Participant ID string (e.g., `brian1control`) |

> **Critical**: The `participant` field naming convention should follow the pattern `{name}{session}{group}`, where `group` is either `control` or `neurofeedback`. The pipeline uses this field to assign group membership for group-level statistics.

---

## 6. Step 1: Raw EEG to Trial Data (`eeg_to_trials.py`)

### 6.1 Purpose

`eeg_to_trials.py` takes the raw OpenBCI recording and produces a structured per-trial CSV by dividing the recording into evenly spaced epochs and extracting spectral features from each epoch. Because no behavioral event markers are embedded in this recording, trial onsets are assigned programmatically at uniform intervals.

### 6.2 Command

```bash
python3 eeg_to_trials.py \
  --input "/path/to/OpenBCI-RAW-YYYY-MM-DD_HH-MM-SS.txt" \
  --output trials_output.csv \
  --participant brian1control \
  --n_trials 80 \
  --n_blocks 4
```

### 6.3 Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | Yes | — | Path to raw OpenBCI or BrainFlow `.txt` file |
| `--output` | No | `trials_output.csv` | Output CSV filename |
| `--format` | No | `openbci` | Input format: `openbci` or `brainflow` |
| `--participant` | No | `unknown` | Participant ID written to the CSV |
| `--n_trials` | No | 80 | Total number of trials to generate |
| `--n_blocks` | No | 4 | Number of blocks to divide trials across |
| `--epoch_duration` | No | (script default) | Duration of each epoch in seconds |
| `--epoch_offset` | No | (script default) | Time offset from trial onset for epoch start |
| `--seed` | No | (script default) | Random seed for stimulus letter generation |

### 6.4 Expected Output (Confirmation)

A successful run will print:

```
Loaded 28951 samples | 8 ch | 115.6s @ 250 Hz
Saved 80 trials → trials_output.csv
```

Followed by a preview of the first five rows.

### 6.5 Common Mistake

Do **not** run `eeg_to_trials.py` without the `--input` argument. The script requires it and will exit with:

```
eeg_to_trials.py: error: the following arguments are required: --input
```

Always pass the full path to the raw recording file.

---

## 7. Step 2: Analysis Pipeline (`eeg_pipeline.py`)

### 7.1 Purpose

`eeg_pipeline.py` reads the trial CSV produced in Step 1, constructs MNE epochs from the embedded beta power timeseries, applies artifact rejection, and computes three primary analyses:

1. **Beta–RT correlation**: Pearson correlation between pre-stimulus beta power and reaction time.
2. **Correct vs. Missed beta power**: Independent-samples t-test comparing beta power on correct and incorrect trials.
3. **Beta power by block**: Mean beta power across each experimental block, revealing trends over time.

### 7.2 Command

```bash
python3 eeg_pipeline.py
```

> The script reads `trials_output.csv` from the working directory by default. Ensure the terminal is `cd`'d into the project folder, or modify the input path constant at the top of the script.

### 7.3 Key Parameters (Internal)

| Parameter | Current Value | Description |
|-----------|--------------|-------------|
| Artifact threshold | 1,000,000 µV | Extremely permissive; intended for simulated data. Tighten to ~100–150 µV for live recordings. |
| Epoch window | −200 to +500 ms | Pre-stimulus baseline plus response window |
| Beta band | 13–30 Hz | Standard beta definition |
| Alpha band | 8–13 Hz | Standard alpha definition |
| Theta band | 4–8 Hz | Standard theta definition |

### 7.4 Expected Console Output

```
[LOAD] 80 trials loaded from trials_output.csv
[EPOCH] 38 epochs × 179 samples (-200 to 500 ms)
[CLEAN] 38 / 38 epochs passed artifact rejection (threshold = 1000000.0 µV)
[ANALYZE] Beta–RT correlation: r=-0.093, p=0.4113 (n=80)
[ANALYZE] Correct beta=-9.93 dB, Missed beta=-9.50 dB, p=0.8144
[ANALYZE] Beta by block: {1: -6.84, 2: -11.94, 3: -11.42, 4: -9.41}
[GROUP] No neurofeedback data yet — baseline only.
[GROUP] Control baseline — beta=-9.90 dB, accuracy=93.8%
[VIZ] Saved eeg_pipeline_results.png
[DONE] Pipeline complete.
```

---

## 8. Output Interpretation

### 8.1 Beta–RT Correlation

A **negative** r-value indicates that higher pre-stimulus beta power is associated with **faster** reaction times. This is the expected direction, consistent with the hypothesis that beta power reflects cortical readiness or attentional engagement.

| Result | Interpretation |
|--------|---------------|
| r = −0.304, p = 0.0106 (eeg_trial_data.csv run) | Statistically significant negative correlation; beta predicts RT |
| r = −0.093, p = 0.4113 (trials_output.csv run) | Non-significant; no reliable beta–RT relationship in this dataset |

The discrepancy between runs reflects differences in the two input datasets and warrants investigation (see Section 9).

### 8.2 Correct vs. Missed Beta Power

A higher beta power on correct trials relative to missed trials supports a signal-detection interpretation of beta's role. Non-significant results (p > 0.05) suggest the current dataset lacks sufficient power or that the trial generation method introduces noise.

### 8.3 Beta by Block

Increasing beta power across blocks may indicate fatigue-related arousal changes or learning effects. Block 4 showing the highest value (14.01 dB) in the `eeg_trial_data.csv` run is noteworthy.

### 8.4 Group-Level Output

When neurofeedback session data is added (a second participant CSV with a `neurofeedback` identifier), the pipeline will output a comparison against the control baseline. Currently only control baseline data exists.

---

## 9. Common Errors and Remediation

| Error | Cause | Fix |
|-------|-------|-----|
| `error: the following arguments are required: --input` | `eeg_to_trials.py` run without arguments | Always supply `--input /path/to/file.txt` |
| `[Errno 2] No such file or directory` with a concatenated path | Two commands accidentally merged into one shell line (e.g., running the pipeline script and `eeg_to_trials.py` as a single command) | Run each script as a separate command in separate terminal lines |
| `trials_output.csv` missing | `eeg_to_trials.py` not yet run | Complete Step 1 before Step 2 |
| Epoch count lower than trial count | Not all trials fall within the recording window | Check `--n_trials` relative to recording duration; reduce trial count or lengthen recording |
| Negative dB beta values | Beta power is below the broadband reference level | This is physically valid; it indicates the beta band is attenuated relative to broadband power |

> **Regarding the concatenated-path error**: Near the bottom of the terminal log, the command `/opt/homebrew/bin/python3 "...eeg_pipeline.py"python3 eeg_to_trials.py ...` was submitted as a single string. This caused the interpreter to search for a file literally named `eeg_pipeline.pypython3`, which does not exist. Each script invocation must be submitted as its own command.

---

## 10. Glossary

| Term | Definition |
|------|-----------|
| **Beta power** | Spectral power in the 13–30 Hz EEG frequency band, associated with active cognition, motor readiness, and sustained attention |
| **Epoch** | A time-locked segment of EEG data surrounding a trial event |
| **Artifact rejection** | The process of discarding epochs that exceed an amplitude threshold, indicating contamination by movement or electrical noise |
| **dB (decibel)** | A logarithmic unit used here to express band power relative to a reference; negative dB indicates power below the reference |
| **Welch PSD** | A method for estimating the power spectral density of a signal by averaging periodograms of overlapping segments |
| **Neurofeedback** | A training paradigm in which a participant receives real-time feedback about their own EEG signal and learns to modulate it |
| **MNE** | MNE-Python, an open-source library for EEG and MEG data processing |
| **BrainFlow** | An open-source library for acquiring biosignals from devices including OpenBCI boards |

---

*This document is part of the CruX EEG Neurofeedback BCI project SOP chain. The next document in the chain covers the `neuroflow-eeg-analytics` React frontend and its integration with this pipeline's CSV output.*
