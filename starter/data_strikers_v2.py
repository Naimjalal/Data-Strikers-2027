
"""Data Strikers V2 - two-stage possession prediction.

Uses:
1. Player and ball tracking
2. Automatic Smart Tagging
3. A separate dead-ball detector

No ground-truth labels are used.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, require
from starter.baseline import build, write_submission

FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]

# Initial design assumptions, not tuned on ground truth.
WINDOW = 50
TAG_WEIGHT = 0.25
STATIONARY_SECONDS = 2.0
STATIONARY_SPEED = 0.4
PLAYER_DISTANCE = 2.5

ATTACKING_CODES = {
    "BUILD UP",
    "PROGRESSION",
    "CREATING CHANCES",
    "BALL IN FINAL THIRD",
    "ATTACKING TRANSITION",
}


def moving_sum(values, window):
    cumulative = np.concatenate(
        ([0.0], np.cumsum(values, dtype=np.float64))
    )

    frames = np.arange(len(values))
    low = np.maximum(0, frames - window)
    high = np.minimum(len(values), frames + window + 1)

    return cumulative[high] - cumulative[low]


def read_ball_signals():
    """Read automatic ball and player observations."""
    meta, raw = require(
        ["atd_meta", "atd_raw"], GAME
    )

    visible = np.zeros(N_FRAMES, dtype=bool)
    stationary = np.zeros(N_FRAMES, dtype=bool)
    outside = np.zeros(N_FRAMES, dtype=bool)
    tracked_players = np.zeros(N_FRAMES, dtype=np.int16)

    pitch_length, pitch_width = GAMES[GAME]["pitch"]

    with open(raw) as handle:
        for line in handle:
            parts = line.strip().split(":")

            if len(parts) != 3:
                continue

            frame = int(parts[0]) - 1
            if not 0 <= frame < N_FRAMES:
                continue

            players = parts[1].split(";")

            tracked_players[frame] = sum(
                bool(p) and not p.startswith("N")
                for p in players
            )

            ball = parts[2].split(",")

            if len(ball) < 3 or ball[0].startswith("N"):
                continue

            x, y, speed = map(float, ball[:3])
            # Recognize a ball when its position is available.
            if not np.isfinite([x, y]).all():
                continue
            visible[frame] = True

            # Position beyond the pitch boundaries.
            outside[frame] = (
                x < 0 or x > 1 or y < 0 or y > 1
            )

            # Nearest visible player to the ball.
            nearest = float("inf")

            for player in players:
                values = player.split(",")

                if len(values) < 2 or values[0].startswith("N"):
                    continue

                try:
                    px = float(values[0])
                    py = float(values[1])
                except ValueError:
                    continue

                if not np.isfinite([px, py]).all():
                    continue

                distance = np.hypot(
                    (px - x) * pitch_length,
                    (py - y) * pitch_width
                )
                nearest = min(nearest, distance)

            stationary[frame] = (
                np.isfinite(speed)
                and speed <= STATIONARY_SPEED
                and nearest >= PLAYER_DISTANCE
                )

    print("Visible ball frames:", int(visible.sum()))

    return visible, stationary, outside, tracked_players


def load_smart_evidence():
    path = require(["smart_tagging"], GAME)[0]
    root = ET.parse(path).getroot()

    evidence = np.zeros((2, N_FRAMES), dtype=np.float32)
    used = 0

    for instance in root.iter("instance"):
        code = (instance.findtext("code") or "").upper()

        if code not in ATTACKING_CODES:
            continue

        labels = {
            label.findtext("group"): label.findtext("text")
            for label in instance.findall("label")
        }

        team_name = (labels.get("Team") or "").lower()

        if team_name == "barcelona":
            team = 0
        elif team_name in ("atletic club", "athletic club"):
            team = 1
        else:
            continue

        start = int(round(float(instance.findtext("start")) * FPS))
        end = int(round(float(instance.findtext("end")) * FPS))

        start = max(0, min(start, N_FRAMES))
        end = max(0, min(end, N_FRAMES))

        if end > start:
            evidence[team, start:end] += TAG_WEIGHT
            used += 1

    print("Smart Tagging events used:", used)
    return evidence


def detect_dead_ball(visible, stationary, outside, players):
    """Conservative automatic dead-ball candidates.

    Missing ball observations alone do not mean dead ball.
    """
    dead = outside.copy()

    # Require sustained stationary-ball evidence.
    minimum = int(STATIONARY_SECONDS * FPS)

    start = None

    for i in range(N_FRAMES + 1):
        active = i < N_FRAMES and stationary[i]

        if active and start is None:
            start = i

        if not active and start is not None:
            if i - start >= minimum:
                dead[start:i] = True
            start = None

    # Long tracking gaps with almost no players are
    # another indication that active play is absent.
    sparse = players < 5
    start = None

    for i in range(N_FRAMES + 1):
        active = i < N_FRAMES and sparse[i]

        if active and start is None:
            start = i

        if not active and start is not None:
            if i - start >= 2 * FPS:
                dead[start:i] = True
            start = None

    # print("Dead-ball candidate frames:", int(dead.sum()))
    print("Outside-pitch frames:", int(outside.sum()))
    print("Stationary-ball frames:", int(stationary.sum()))
    print("Sparse-player frames:", int((players < 5).sum()))
    print("Dead-ball candidate frames:", int(dead.sum()))

    graded = np.zeros(N_FRAMES, dtype=bool)

    graded[11185:90652] = True
    graded[110808:190198] = True

    print(
        "Dead-ball candidates during graded frames:",
        int(np.sum(dead & graded))
    )

    print(
        "Outside-pitch frames during graded frames:",
        int(np.sum(outside & graded))
    )

    print(
        "Sparse-player frames during graded frames:",
        int(np.sum((players < 5) & graded))
    )





    return dead


def predict():
    print("\nDATA STRIKERS V2")
    print("================")

    codes = build()

    visible, stationary, outside, players = (
        read_ball_signals()
    )

    tag_evidence = load_smart_evidence()

    home = (codes == 0).astype(np.float32)
    away = (codes == 1).astype(np.float32)

    home += tag_evidence[0]
    away += tag_evidence[1]

    home_score = moving_sum(home, WINDOW)
    away_score = moving_sum(away, WINDOW)

    predictions = np.full(
        N_FRAMES, "D", dtype="<U1"
    )

    known = (home_score + away_score) > 0

    predictions[known] = np.where(
        home_score[known] >= away_score[known],
        "H", "A"
    )

    # Dead-ball evidence takes priority over team evidence.
    dead = detect_dead_ball(
        visible, stationary, outside, players
    )

    predictions[dead] = "D"

    # Publicly documented ungraded video periods.
    predictions[:11185] = "X"
    predictions[90652:110808] = "X"
    predictions[190198:] = "X"

    output = os.path.join(
        ROOT,
        "evaluation",
        "results",
        "data_strikers_v2.csv"
    )

    write_submission(output, predictions)

    print("\nFile created:", output)

    for label in ("H", "A", "D", "X"):
        print(label, int(np.sum(predictions == label)))

    print("Total frames:", len(predictions))


if __name__ == "__main__":
    predict()
