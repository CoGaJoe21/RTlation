"""Satzweises Aufteilen des Textes vor der Sprachausgabe (workers/base.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

from base import split_sentences  # noqa: E402


def test_single_sentence_stays_whole():
    text = "Tomorrow I'm taking the train to Zurich, where I'll meet an old friend."
    assert split_sentences(text) == [text]


def test_two_sentences_are_split():
    first = "Good day, I am currently testing a new program."
    second = "It is supposed to translate my voice in real time."
    assert split_sentences(f"{first} {second}") == [first, second]


def test_short_pieces_stay_with_their_neighbour():
    assert split_sentences("Yes. I will take the train to Zurich tomorrow.") == [
        "Yes. I will take the train to Zurich tomorrow."
    ]
    assert split_sentences("I will take the train to Zurich tomorrow. Really!") == [
        "I will take the train to Zurich tomorrow. Really!"
    ]


def test_chinese_sentences_are_split_without_spaces():
    first = "明天我将乘坐火车前往苏黎世，在那里与一位老朋友见面，然后我们一起去吃饭。"
    second = "它的目标是实时将我的声音翻译成另一种语言，这样别人就能听懂我说的话。"
    assert split_sentences(first + second) == [first, second]


def test_empty_text_does_not_crash():
    assert split_sentences("") == [""]
