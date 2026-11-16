# Data Strikers
## Technical Report: More Than a Hack 2027
### Initial Challenge: Football Possession Prediction

**Team:** Data Strikers  
**Submission algorithm:** Data Strikers V3b  
**Match:** FC Barcelona vs Athletic Club  
**Competition:** More Than a Hack 2027

---

## 1. Project Objective

The objective of this challenge is to estimate which team controls the ball for every frame of the provided tactical football video.

Our algorithm generates a prediction for all **199,319 frames** using four possible labels:

| Label | Meaning |
|---|---|
| H | Home team possession: Barcelona |
| A | Away team possession: Athletic Club |
| D | Dead ball or out-of-play situation |
| X | Unknown or outside the graded match intervals |

The principal competition evaluation metric is **Spell F1**, which measures the quality of predicted possession segments across several temporal intersection-over-union thresholds.

Our objective is to produce a deterministic and reproducible possession prediction system using automatic tracking and tactical event information.

---

## 2. Overview of Our Approach

Data Strikers V3b is a rule-based possession estimation system developed in Python.

It combines:

1. Automatic Tracking Data (ATD).
2. Nearest-player-to-ball estimates.
3. Automatic Smart Tagging events.
4. Temporal smoothing.
5. Possession continuity using hysteresis.
6. Tracking-based dead-ball detection.

The system does not require training a machine learning model.

All possession labels are generated automatically by the algorithm.

No manual possession annotations from the provided match were used to train, tune or correct the submitted predictions.

---

## 3. Input Data

The algorithm uses the following competition-provided files:

### 3.1 Automatic Tracking Data

- `BAR-ATH_tactical_atd.xml`
- `BAR-ATH_tactical_atd.txt`

These files contain player-tracking information, team assignments, ball coordinates when available, and tracking metadata.

The tracking records are aligned to the tactical video timeline.

Each tracking record uses a one-based frame number, converted to the zero-based submission frame number using:

`video_frame = frame_id - 1`

Player coordinates are normalized relative to the pitch dimensions.

The XML metadata identifies the home and away teams and associates tracking slots with teams.

Tracking slots are treated as temporary tracklets rather than guaranteed permanent player identities.

### 3.2 Automatic Smart Tagging

- `BAR-ATH_tactical_smart_tagging.xml`

This file contains automatically generated tactical events.

Our implementation uses selected attacking events, including:

- Build up
- Progression
- Creating chances
- Ball in final third
- Attacking transition

These events provide supporting evidence about attacking activity but are not treated as definitive possession labels.

The V3b implementation also examines set-piece events to identify possible stoppage candidates. Those candidates are diagnostic and are not directly combined with the final dead-ball override.

### 3.3 Assets Investigated but Not Used in the Final Algorithm

We investigated the potential use of the tactical video, homography data and a pretrained YOLO object detector.

The preliminary ball-detection approach was not sufficiently reliable to justify including it in the final V3b submission.

The submitted implementation does not depend on YOLO, OpenCV, PyTorch, the tactical video file or the homography file.

---

## 4. Possession Estimation

### 4.1 Nearest-Player Evidence

For frames with available ball coordinates, the baseline tracking method estimates which team's player is closest to the ball.

Distances are calculated using the pitch dimensions so that normalized horizontal and vertical coordinates are converted into comparable physical distances.

This produces an initial home-team or away-team signal.

When the ball is unavailable, this stage cannot establish which player is closest to it.

### 4.2 Smart Tagging Evidence

The algorithm incorporates automatically generated attacking event information as additional evidence.

Selected events contribute evidence for the team associated with the event over its annotated frame interval.

These signals support the tracking estimates but are not considered ground-truth possession.

### 4.3 Temporal Smoothing

Possession evidence is aggregated over a temporal window to reduce rapid changes caused by noisy tracking.

The selected implementation uses:

`WINDOW = 50`

At 25 frames per second, this is a centered window extending approximately two seconds before and after each frame.

Temporal smoothing makes predictions less sensitive to isolated tracking errors.

### 4.4 Possession Continuity

The algorithm applies hysteresis to reduce unnecessary changes between teams.

The current possession is maintained unless the opposing team's evidence exceeds the current team's evidence by a specified margin.

The implementation uses:

`SWITCH_MARGIN = 0.08`

This margin is applied relative to the combined possession evidence.

These parameters are algorithm-design settings and were not selected by optimizing against the provided game's public ground-truth labels.

When neither team has available evidence, the V3b implementation resets its current possession state and leaves the prediction as `D` unless another rule applies.

This behavior is an important limitation because missing evidence does not necessarily imply that play has stopped.

---

## 5. Dead-Ball Detection and Possession Rules

The system uses automatic tracking observations to identify possible dead-ball situations.

### 5.1 Outside-Pitch Ball

When available ball coordinates indicate that the ball is outside the normalized pitch boundaries, the detector marks a dead-ball candidate.

### 5.2 Sustained Stationary Ball

The detector identifies sustained stationary-ball evidence as another possible indication of a stoppage.

### 5.3 Sparse Player Tracking

The current V3b detector also identifies intervals with fewer than five tracked players for a sustained period.

This is a heuristic rather than proof of dead ball, because low player-tracking coverage may reflect tracking limitations.

### 5.4 Set Pieces and Restarts

Automatic set-piece events are examined to identify possible stoppages associated with missing-ball observations.

However, V3b does not automatically label an entire set-piece event as dead ball.

Its final override uses the tracking-based dead-ball detector.

### 5.5 Loose Balls and Challenges

The algorithm does not contain a dedicated physical ball-control model for loose balls or contested challenges.

These situations are estimated using nearest-player evidence, temporal smoothing and possession continuity.

As a result, the algorithm may attribute possession to a nearby player even when neither team has established control.

### 5.6 Injuries, VAR, Cooling Breaks and Added Time

The model does not explicitly classify the reason for a stoppage.

Injuries, VAR reviews, substitutions and cooling breaks may result in dead-ball periods, but the system does not rely on manually identified incidents from the provided match.

Added time is treated as part of the relevant match period. It is not automatically classified as dead ball.

---

## 6. Training Data and External Labels

The submitted V3b method is rule-based.

**No supervised training dataset was used.**

No human-labeled data from the provided match was used to train, tune or validate model parameters.

The publicly provided possession labels were used only for permitted pipeline diagnostics and not as a parameter-optimization dataset.

The submitted prediction sequence is generated by the documented algorithm using the licensed automatic match assets.

---

## 7. Output and Reproducibility

The algorithm creates a CSV containing:

`frame,label`

The file contains exactly **199,319 data rows**, numbered from frame 0 through frame 199,318.

Predictions outside the configured graded periods are marked `X`.

The submitted reference output contains:

| Label | Number of frames |
|---|---:|
| H | 72,317 |
| A | 48,714 |
| D | 37,826 |
| X | 40,462 |
| Total | 199,319 |

The final CSV was validated using the organizers' official submission validator.

The output was reproduced from a separate clean source-code folder using the same licensed input assets and Python environment.

Both runs produced the same SHA-256 checksum:

`6dd6c00f42e55a622caf13db1644c08ffe22b496d93f671538e4c71dc04be01a`

This demonstrates byte-identical local reproducibility under the tested environment.

---

## 8. Software Requirements

Tested environment:

- Python 3.13.5
- NumPy 2.5.4
- Pandas 3.1.0

The main prediction algorithm requires NumPy and the included Python source files.

Pandas is used by the official CSV validation and evaluation utilities.

No GPU is required.

### Required Data Directory

The licensed input files must be placed in:

`data/tactical/`

The required files are:

- `BAR-ATH_tactical_atd.xml`
- `BAR-ATH_tactical_atd.txt`
- `BAR-ATH_tactical_smart_tagging.xml`

The licensed match assets are not redistributed in the public code repository.

### Installation

From the repository root:

```bash
py -m pip install -r requirements-data-strikers.txt
```

### Generate Predictions

```bash
py starter/data_strikers_v3b.py
```

The program writes:

`evaluation/results/data_strikers_v3b.csv`

### Validate the Submission

```bash
py evaluation/validate_submission.py evaluation/results/data_strikers_v3b.csv
```

The competition submission file is named `Data_Strikers.csv` and contains the same validated prediction data.

---

## 9. Limitations

The principal limitations of the current system are:

**Missing ball tracking:** Ball coordinates are unavailable during substantial portions of the match. The algorithm cannot directly estimate nearest-player possession in these intervals.

**Dead-ball uncertainty:** V3b sometimes classifies missing possession evidence as `D` even when there is no explicit tracking-based stoppage signal.

**Long tracking gaps:** During long intervals without ball coordinates, the algorithm may fail to identify genuine possession changes.

**Nearest-player ambiguity:** The player closest to the ball is not always the player controlling it, especially during challenges, interceptions and loose-ball situations.

**Automatic event uncertainty:** Tactical event annotations may cover broad intervals and are not guaranteed possession boundaries.

**Tracking identity continuity:** Player-tracking slots may change or disappear, making stable individual-player movement analysis difficult.

**Stoppage recognition:** The algorithm does not explicitly distinguish VAR reviews, injuries, substitutions or other causes of interrupted play.

**External generalization:** Full-match prediction accuracy is not known, and reproducibility on an independently configured environment has not yet been verified.

---

## 10. Future Improvements

Given additional development time, we would investigate:

1. A more reliable automatic ball detector, evaluated on permitted independent footage.
2. Independent labeled possession data from other matches to measure generalization.
3. A confidence-based dead-ball classifier using player movement, ball visibility and restart evidence.
4. Improved possession-change detection using spatial and temporal features.
5. Better handling of tracking gaps without automatically assuming dead ball or extending possession indefinitely.

Any improvements would remain fully algorithmic and reproducible.

---

## 11. Conclusion

Data Strikers developed a deterministic football possession prediction pipeline combining automatic tracking data, Smart Tagging, temporal smoothing and possession continuity.

The system generates a complete, valid CSV submission for the supplied tactical video and has demonstrated byte-identical reproducibility in the tested local environment.

The solution is submitted as an automatic possession estimator with documented assumptions and limitations.

**Team: Data Strikers**  
**More Than a Hack 2027, Initial Challenge**