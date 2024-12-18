import pytest

from corppa.poetry_detection.identify_poems import searchable_text

search_texts = [
    # strip punctuation and remove accents
    ("So many ǎs lōve mě, ănd ūse mě ǎright", "So many as love me and use me aright"),
    # remove newlines, punctuation, omit single - or ' in middle of word
    (
        """the hour when | rites un | -holy
ed each | Paynim | voice to | pray'r""",
        "the hour when rites unholy ed each Paynim voice to prayr",
    ),
    # strip outer whitespace; handle long s
    (
        """  Thoſe darts whoſe points make gods adore
His might, and deprecate his power.   """,
        "Those darts whose points make gods adore His might and deprecate his power",
    ),
    # metric notation that splits words
    (
        "nearly napping, | sudden | -ly there came a | tapping ... some visit | -or",
        "nearly napping suddenly there came a tapping some visitor",
    ),
]


@pytest.mark.parametrize("input,expected", search_texts)
def test_searchable_text(input, expected):
    assert searchable_text(input) == expected
