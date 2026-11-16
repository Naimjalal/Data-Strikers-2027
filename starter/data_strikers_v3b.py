
"""DATA STRIKERS V3
Football possession prediction using tracking,
Smart Tagging and possession continuity.

No ground-truth labels are used.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES
from starter.baseline import build, write_submission
from starter.load_smart_tagging import load
from starter.data_strikers_v2 import (
    read_ball_signals,
    load_smart_evidence,
    detect_dead_ball,
    moving_sum,
)

FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]

# Initial football-based design assumptions.
# These have NOT been tuned against public ground truth.
WINDOW = 50
SWITCH_MARGIN = 0.08
MIN_MISSING_SECONDS = 2.0


def set_piece_evidence(visible, players):
    """Identify possible stoppages around automatic
    set-piece events using missing-ball evidence.

    An event interval alone does not prove dead ball.
    """

    events = load()
    candidates = np.zeros(N_FRAMES, dtype=bool)

    minimum_gap = int(
        MIN_MISSING_SECONDS * FPS
    )

    count = 0

    for event in events:

        if (event["code"] or "").upper() != "SET PIECES":
            continue

        count += 1

        start = max(
            0, event["start_frame"]
        )

        end = min(
            N_FRAMES, event["end_frame"]
        )

        # Search for sustained ball-tracking gaps
        # within the automatic event interval.
        gap_start = None

        for frame in range(start, end + 1):

            missing = (
                frame < end
                and not visible[frame]
            )

            if missing and gap_start is None:
                gap_start = frame

            if not missing and gap_start is not None:

                length = frame - gap_start

                if length >= minimum_gap:
                    candidates[gap_start:frame] = True

                gap_start = None

    # Require that players remain tracked, avoiding
    # classifying unrelated camera/tracking outages.
    candidates &= (players >= 5)

    print("Set-piece events examined:", count)
    print(
        "Set-piece stoppage candidates:",
        int(candidates.sum())
    )

    return candidates


def stable_possession(home_score, away_score):
    """Apply hysteresis so uncertain brief changes
    do not immediately switch possession.

    The team changes only when opposing evidence
    exceeds the current team's evidence by a margin.
    """

    predictions = np.full(
        N_FRAMES, "D", dtype="<U1"
    )

    current = None

    for frame in range(N_FRAMES):

        home = home_score[frame]
        away = away_score[frame]

        total = home + away

        if total <= 0:
            current = None
            continue

        margin = total * SWITCH_MARGIN

        if current is None:
            current = "H" if home >= away else "A"

        elif current == "H":

            if away > home + margin:
                current = "A"

        elif current == "A":

            if home > away + margin:
                current = "H"

        predictions[frame] = current

    return predictions


def predict():

    print("\nDATA STRIKERS V3")
    print("===================")

    # 1. Tracking-based team evidence
    codes = build()

    # 2. Ball and player signals
    visible, stationary, outside, players = (
        read_ball_signals()
    )

    # 3. Automatic attacking evidence
    tags = load_smart_evidence()

    home = (codes == 0).astype(np.float32)
    away = (codes == 1).astype(np.float32)

    home += tags[0]
    away += tags[1]

    # 4. Smooth observations
    home_score = moving_sum(home, WINDOW)
    away_score = moving_sum(away, WINDOW)

    # 5. Stable possession estimation
    predictions = stable_possession(
        home_score, away_score
    )

    # 6. Existing V2 dead-ball detector
    tracking_dead = detect_dead_ball(
        visible, stationary, outside, players
    )

    # 7. Additional set-piece evidence
    restart_dead = set_piece_evidence(
        visible, players
    )

    # dead = tracking_dead | restart_dead
    # Set-piece gaps are diagnostic evidence only.
    # Do not automatically classify them as dead ball.
    dead = tracking_dead.copy()
    graded = np.zeros(N_FRAMES, dtype=bool)
    graded[11185:90652] = True
    graded[110808:190198] = True

    print("\nDEAD-BALL DETECTOR ANALYSIS")
    print("---------------------------")

    print(
        "Tracking detector:",
        int(np.sum(tracking_dead & graded))
)

    print(
        "Set-piece detector:",
        int(np.sum(restart_dead & graded))
)

    print(
        "Both detectors agree:",
    int(np.sum(tracking_dead & restart_dead & graded))
)

    print(
        "Tracking only:",
        int(np.sum(tracking_dead & ~restart_dead & graded))
)

    print(
        "Set-piece only:",
        int(np.sum(restart_dead & ~tracking_dead & graded))
)

    print(
        "Either detector:",
        int(np.sum(dead & graded))
        )

    predictions[dead] = "D"

    # Publicly documented ungraded periods.
    predictions[:11185] = "X"
    predictions[90652:110808] = "X"
    predictions[190198:] = "X"

    # 8. Write complete CSV
    output = os.path.join(
        ROOT,
        "evaluation",
        "results",
        "data_strikers_v3b.csv"
    )

    write_submission(output, predictions)

    print("\nV3 predictions generated")
    print("File:", output)

    for label in ("H", "A", "D", "X"):
        print(
            label,
            int(np.sum(predictions == label))
        )

    print("Total frames:", len(predictions))


if __name__ == "__main__":
    predict()
