"""The decision prompt text (no model or tokenizer needed)."""
from jevision.prompts import YES_NO_OPTIONS, user_text


def test_choice_options_are_lettered():
    text = user_text("How many?", ["2", "3", "4"])
    assert text.endswith("options:\nA) 2: 2\nB) 3: 3\nC) 4: 4")
    assert text.startswith("record: [image 1]\nquestion: How many?")


def test_yes_no_options():
    assert user_text("Is it red?").endswith("options:\n" + YES_NO_OPTIONS)
