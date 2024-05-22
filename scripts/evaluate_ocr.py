import os
import sys
import spacy
import csv
import json
# import syntok.segmenter as segmenter

from helper import open_jsonl
from ocr_helper import clean_chars
from tqdm import tqdm
from lingua import LanguageDetectorBuilder


class OCREvaluator:
    """
    Object for OCR evaluations.
    """

    def __init__(self):
        disable_defaults = [
            "tok2vec",
            "tagger",
            "parser",
            "attribute_ruler",
            "lemmatizer",
            "ner",
        ]
        self.spacy_model = spacy.load("en_core_web_lg", disable=disable_defaults)
        self.vocab = self.__init_vocab()
        self.lang_detector = LanguageDetectorBuilder.from_all_languages().build()
        self.languages = {lang.name: lang for lang in self.lang_detector._languages}

    def __init_vocab(self):
        """
        Initialize reference vocabulary that will be used for
        dictionary lookup measurement
        """
        # Filter vocab further by removing words without static embeddings
        vocab = []
        for w in self.spacy_model.vocab.strings:
            for t in self.spacy_model(w):
                if not t.is_oov:
                    vocab.append(t.text.lower())
        return set(vocab)

    def dict_lookup(self, text, min_chars=3):
        """
        Return percentage of (lower-cased, min_char+ length) tokens in
        reference dictionary; if there are no such tokens, return -1
        """
        num = 0
        den = 0
        for token in self.spacy_model(text):
            if len(token.text) >= min_chars:
                if token.text.lower() in self.vocab:
                    num += 1
                den += 1
        return num / den if den > 0 else -1

    # TODO: Consider combinining the two language detection methods for speed
    def detect_language(self, text, lang):
        """
        Return the confidence value for detecting a language (lang) for the
        input text
        """
        if lang.upper() not in self.languages:
            print(f"ERROR: Unsupported language '{lang}'")
            raise ValueError
        return self.lang_detector.compute_language_confidence(
            text, self.languages[lang.upper()]
        )

    def detect_top_languages(self, text, top_k=3):
        """
        Determine the top k (default 3) most detected languages for the input text.
        Return: list of tuples (language: str, confidence value: float)
        """
        if top_k > len(self.languages):
            print(f"ERROR: 'top-k' is too high (>{len(self.languages)})")
            raise ValueError
        cvals = self.lang_detector.compute_language_confidence_values(text)
        result = [("N/A", 0)] * top_k
        for i in range(top_k):
            cv = cvals[i]

            # If confidence value equals 0.0, there's nothing further to update
            if cv.value == 0.0:
                return result

            result[i] = (cv.language.name, cv.value)

            # Check for unambiguous detection
            if i == 0 and cv.value == 1.0:
                return result
        return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: [input jsonl] [output tsv]")
        sys.exit(1)

    in_jsonl = sys.argv[1]
    out_tsv = sys.argv[2]

    if not os.path.isfile(in_jsonl):
        print(f"ERROR: '{in_jsonl}' does not exist")
        sys.exit(1)
    if os.path.isfile(out_tsv):
        print(f"ERROR: '{out_tsv}' exists")
        sys.exit(1)

    # Initial evaluator
    evaluator = OCREvaluator()

    # Output tsv fields
    field_names = [
        "page_id",
        "work_id",
        "dict_lookup",
        "is_eng",
        "lang 1",
        "conf 1",
        "lang 2",
        "conf 2",
        "lang 3",
        "conf 3",
    ]

    # Get line count for tqdm
    n_lines = sum([1 for line in open_jsonl(in_jsonl, mode="rb")])
    with open(out_tsv, mode="w", newline="") as file_handler:
        writer = csv.DictWriter(
            file_handler, dialect="excel-tab", fieldnames=field_names
        )
        writer.writeheader()

        with open_jsonl(in_jsonl) as reader:
            for line in tqdm(reader, total=n_lines):
                page = json.loads(line)
                text = clean_chars(page.get("text", ""))
                score = evaluator.dict_lookup(text)
                entry = {
                    "page_id": page["id"],
                    "work_id": page["work_id"],
                    "dict_lookup": f"{score:.4g}" if score > 0 else "N/A",
                }

                # English Language Detection
                conf = evaluator.detect_language(text, "english")
                entry["is_eng"] = f"{conf:.4g}"

                # Top-3 Language Detection
                for i, tup in enumerate(evaluator.detect_top_languages(text)):
                    lang, conf = tup
                    entry[f"lang {i+1}"] = lang
                    entry[f"conf {i+1}"] = f"{conf:.4g}"
                writer.writerow(entry)
