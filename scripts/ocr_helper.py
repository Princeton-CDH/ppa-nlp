"""
Library of OCR-related auxiliary methods for stand-alone scripts
"""

import ftfy


_char_conversion_map = {"ſ": "s"}
_char_translation_table = str.maketrans(_char_conversion_map)


def clean_chars(text):
    """
    Initial cleaning of text focused on characters
    """
    result = ftfy.fix_text(
        text,
        unescape_html=False,
        fix_encoding=False,
        normalization="NFC",
        explain=False,
    )
    result = result.translate(_char_translation_table)
    return result
