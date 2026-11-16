# Data Strikers
## More Than a Hack 2027 - Initial Challenge

### Objective
Predict football possession for every frame of the
Barcelona vs Athletic Club tactical video.

Labels:
- H: Barcelona possession
- A: Athletic Club possession
- D: Dead ball
- X: Unknown or outside evaluated periods

### Our approach
Our algorithm combines:
1. Automatic Tracking Data (ATD)
2. Nearest-player-to-ball estimates
3. Automatic Smart Tagging evidence
4. Temporal smoothing and possession continuity
5. Tracking-based dead-ball detection

The algorithm does not use human-labelled possession data
from the provided match.

### Requirements
- Python 3.13.5
- NumPy 2.5.4
- Pandas 3.1.0

Install dependencies:

    py -m pip install -r requirements-data-strikers.txt

### Required input data
Obtain the licensed match data from the competition
organizers and place these files in data/tactical/:

- BAR-ATH_tactical_atd.xml
- BAR-ATH_tactical_atd.txt
- BAR-ATH_tactical_smart_tagging.xml

The match data is not included in this public repository.

### Generate predictions

From the repository root, run:

    py starter/data_strikers_v3b.py

The generated prediction file is:

    evaluation/results/data_strikers_v3b.csv

### Reproducibility
The reference CSV generated in our development environment
has SHA-256 checksum:

6dd6c00f42e55a622caf13db1644c08ffe22b496d93f671538e4c71dc04be01a

The algorithm is deterministic for the supplied data and
documented environment.

### Limitations
Ball tracking is missing in some intervals. The algorithm
can confuse missing possession evidence with dead-ball
situations, and possession changes during long tracking
gaps may be missed.

### Team
Data Strikers
