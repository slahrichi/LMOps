"""Generate a tiny synthetic two-column parquet (content, teacher_response) for the
GAD smoke run. `content` is a chat message list (apply_chat_template is called on it);
`teacher_response` is a plain string. This validates plumbing only — not the paper's data.
"""
from __future__ import annotations

import os

import pandas as pd

OUT_DIR = os.path.join(
    os.environ.get("GAD_WORK", os.path.expanduser("~/gad_run")), "data"
)
os.makedirs(OUT_DIR, exist_ok=True)

# A handful of diverse instruction prompts with coherent, "teacher-quality" answers.
PAIRS = [
    ("What is the capital of France?",
     "The capital of France is Paris, its largest city and political, cultural, and economic center, located on the Seine river in the north-central part of the country."),
    ("Explain what a prime number is in one sentence.",
     "A prime number is a whole number greater than 1 that has exactly two distinct positive divisors: 1 and itself."),
    ("Write a short polite email declining a meeting.",
     "Subject: Unable to Attend\n\nHi Alex,\n\nThank you for the invitation. Unfortunately I have a prior commitment at that time and won't be able to join. Could we find an alternative slot later this week? I'd be glad to catch up then.\n\nBest regards,\nSam"),
    ("What causes rain?",
     "Rain forms when water vapor in the atmosphere cools and condenses into droplets around tiny particles; when these droplets grow heavy enough that rising air can no longer hold them, they fall to the ground as precipitation."),
    ("Give three tips for better sleep.",
     "1) Keep a consistent schedule by going to bed and waking at the same time daily. 2) Avoid screens and caffeine in the hours before bed. 3) Keep your bedroom cool, dark, and quiet to support deeper rest."),
    ("Summarize the water cycle.",
     "The water cycle is the continuous movement of water: it evaporates from oceans and lakes, condenses into clouds, falls as precipitation, and then flows through rivers and groundwater back to the sea, repeating endlessly."),
    ("What is 17 times 4?",
     "17 times 4 equals 68, since 17 x 4 = (10 x 4) + (7 x 4) = 40 + 28 = 68."),
    ("Define photosynthesis briefly.",
     "Photosynthesis is the process by which green plants, algae, and some bacteria use sunlight to convert carbon dioxide and water into glucose and oxygen, storing energy in chemical bonds."),
]


def build_rows(n: int):
    rows = []
    for i in range(n):
        q, a = PAIRS[i % len(PAIRS)]
        rows.append({
            "content": [{"role": "user", "content": q}],
            "teacher_response": a,
        })
    return rows


def main():
    train = pd.DataFrame(build_rows(128))
    val = pd.DataFrame(build_rows(16))
    train_path = os.path.join(OUT_DIR, "smoke_train.parquet")
    val_path = os.path.join(OUT_DIR, "smoke_val.parquet")
    train.to_parquet(train_path)
    val.to_parquet(val_path)
    print("wrote", train_path, len(train), "rows")
    print("wrote", val_path, len(val), "rows")
    print("columns:", list(train.columns))
    print("row0 content:", train.iloc[0]["content"])
    print("row0 teacher_response[:80]:", train.iloc[0]["teacher_response"][:80])


if __name__ == "__main__":
    main()
