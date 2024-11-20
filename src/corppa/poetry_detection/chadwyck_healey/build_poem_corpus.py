"""
Convert the Chadwyck Healey corpus into a single .jsonl file
"""

import argparse
import pathlib
import re
import sys
from collections import Counter
from typing import Any

import bs4
import ftfy
import orjsonl
from tqdm import tqdm

# TODO: Revisit this exclusion list
EXCLUDED_POEMS = {
    "Z200677556",  # Weird poem constructed out of out of unordered lists
}
EXCLUDED_EDITIONS = {
    "Z000594240",  # Modern poetry, poem made of unordered lists
    "Z000359125",  # Modern poetry, poem simply containing an image
    "Z000361067",  # Modern poetry, poem composed of images & captions
    "Z000228617",  # Modern poetry, "poem" that's just a note
    "Z000133831",  # Modern potery, shaped poem uses sl tags
    "Z000683334",  # Modern poetry, poem made up of empty lines
    "Z000582255",  # Modern poetry, poem made of unordered lists
    "Z000229980",  # Modern poetry, poem w/ malformed unordered lists
    "Z000303274",  # Modern poetry, poem w/ malformed unordered lists
    "Z000306246",  # Modern poetry, poem w/ malformed unordered lists
    "Z000605113",  # Modern poetry, poem made of unordered list
    "Z000604899",  # Modern poetry, poem that's a figure
}


# Should this be moved to path utils?
def get_poem_subdir(poem_id):
    """
    Derives the poem's stub directory from its Chadwyck Healey poem (title?) id
    which is assumed to have the following form: [A-Z][0-9]+[a-z]*

    The resulting stub directory has the following form: The leading alphabetic
    characters followed by every third number
        Ex. Z300463864 --> Z348
    """
    pattern = r"([A-Z])(\d+)[a-z]?"
    match = re.fullmatch(pattern, poem_id)
    return f"{match.groups()[0]}{match.groups()[1][::3]}"


class TMLParser:
    SYMBOL_TABLE = {
        "&indent;": "\t",
        "&wblank;": "\u2014\u2014",
        "&lblank;": "\u2014",
        "&grwc;": "\u1ff6",
        "&grst;": "\u03c2",
    }
    TEXT_BLOCKS = {"span", "p"}
    STYLE_TAGS = {"sub", "sup", "it", "u"}
    SKIPPED_TAGS = {
        "edit",
        "stage",
        "break",
        "epilogue",
        "img",
        "castlist",
        "ul",
        "ul2",
    }
    WORKING_DIV_TYPES = {
        "versepara",
        "stanza",
        "line",
        "firstline",
        "conclusion",
        "greek",
        "speaker",
    }
    SKIPPED_DIV_TYPES = {
        "argument",
        "caption",
        "dedication",
        "epigraph",
        "note",
        "preface",
        "signed",
        "trailer",
        "copyright",  # revisit
        "prologue",  # revisit
    }

    def __init__(self, file: pathlib.Path) -> None:
        self.filepath = str(file)
        self.filename = file.stem
        self.log_count = 0
        self.parse_tml(file)

    def __str__(self) -> str:
        return self.filepath

    def print_log(self, message):
        log_message = f"{self}: {message}"
        if self.log_count:
            print(log_message)
        else:
            # tqdm workaround
            print("\n" + log_message)
        self.log_count += 1

    def parse_tml(self, file: pathlib.Path):
        """
        Parse TML file and extract head and body tags
        """
        # 1. Read in raw text
        with open(file, encoding="latin1") as reader:
            text = reader.read()

        # 2. Preprocess text
        ## Hack to fix meta tag for processing header
        text = text.replace("<meta>", "<metadata")
        text = text.replace("</meta>", "</metadata")
        ## Hack attempt to fix ul2/list2 issue
        # if "<list2>" not in text and "<ul2>" in text:
        #    assert "</ul2>" not in text
        #    text = text.replace("</list2>", "</ul2>")
        # TODO: Also deal with mismatched

        ## Replace known problematic tags for processing content
        tag_substitutions = [
            ("figure", r"<figure[^>]*>", ""),
            ("gap", r"<gap>", " "),
            ("div3", r"<div3[^>]*>", ""),
            ("div4", r"<div4[^>]*>", ""),
            ("div5", r"<div5[^>]*>", ""),
            ("caesura", r"<caesura>", ""),
            ("blnkpage", r"<blnkpage>", ""),
            ("xref", r"<xref[^>]*>", ""),
        ]
        for tag, pattern, repl in tag_substitutions:
            assert r"</{tag}>" not in text
            text, n_subs = re.subn(pattern, repl, text)
        if n_subs:
            self.print_log(f"Deleted {n_subs} {tag} tags")

        # 3. Run through Beautiful Soup
        soup = bs4.BeautifulSoup(text, "lxml")
        self.head = soup.head
        self.body = soup.body

    def clean_text(self, text: str) -> str:
        new_text = ftfy.fix_text(text)
        # Complete symbol replacement
        for symbol_str, out_str in self.SYMBOL_TABLE.items():
            new_text = new_text.replace(symbol_str, out_str)
        return new_text.rstrip()

    def _extract_tag_data(self, tag_soup: bs4.Tag) -> dict[str, str]:
        tag_info = {}
        # Attribute handling
        for attr, val in tag_soup.attrs.items():
            tag_info[f"{tag_soup.name}_{attr}"] = val

        # Children tag handling
        for child_tag in tag_soup.find_all(True, recursive=False):
            # Assume child tag has no children tags
            assert child_tag.find(True, recursive=False) is None
            # Handle child tag attributes
            for attr, val in child_tag.attrs.items():
                field_name = f"{child_tag.name}_{attr}"
                assert field_name not in tag_info
                tag_info[field_name] = val
            # Handle child's text
            assert child_tag.name not in tag_info
            tag_info[child_tag.name] = self.clean_text(child_tag.text)
        return tag_info

    def extract_metadata(self) -> dict[str, dict[str, str]]:
        """
        Extract meta data from head tag.
        """
        metadata = {}
        tag_types = Counter(
            [tag.name for tag in self.head.find_all(True, recursive=False)]
        )
        for tag in self.head.find_all(True, recursive=False):
            if tag.name == "body":
                # This shouldn't happen, unless there's a formatting issue
                self.print_log(f"Encountered body tag in head tag, skipping")
                continue
            working_tag = f"{tag.name}"
            if tag_types[tag.name] > 1:
                # Differentiate duplicate tab types by their id
                if "id" in tag.attrs:
                    working_tag = f"{tag.name}_{tag['id']}"
            elif tag.name == "metadata":
                working_tag = "meta"  # Correct for meta tag hack
            if working_tag in metadata:
                raise ValueError(f"{self}: Duplicate metadata tags")
            metadata[working_tag] = self._extract_tag_data(tag)
        return metadata

    def get_bs4_text(self, tag_or_string=bs4.NavigableString | bs4.Tag) -> str:
        """
        Extract text from a BeautifulSoup Tag or NavigableString
        """
        if isinstance(tag_or_string, bs4.NavigableString):
            return str(tag_or_string)
        else:
            tag = tag_or_string
            if tag.name == "div":
                return self.get_div_text(tag)
            elif tag.name in self.SKIPPED_TAGS:
                # Skip these tags
                return ""
            elif tag.name in self.TEXT_BLOCKS | self.STYLE_TAGS:
                # In these cases, simply accumulate children's texts
                text = ""
                for child in tag.children:
                    text += self.get_bs4_text(child)
                if child.name == "p":
                    # For paragraph tag, add ending newline
                    return text + "\n"
                return text
            else:
                # Fallback: Skip and log doing so
                self.print_log(f"Skipping unexpected tag type '{tag.name}'")
                return ""

    def get_div_text(self, div_tag: bs4.Tag) -> str:
        """
        Extract text from a div tag.
        """
        if div_tag.name != "div":
            raise ValueError(f"{div_tag} is not a div tag")
        div_type = div_tag["type"]
        # Handle each div type
        if div_type in self.WORKING_DIV_TYPES:
            text = ""
            for child in div_tag.children:
                text += self.get_bs4_text(child)
            return text + "\n"
        elif div_type in self.SKIPPED_DIV_TYPES:
            # Case 3: Skippable div tags
            return ""
        else:
            # Fallback: Skip and log doing so
            self.print_log(f"Skipping unexpected div type '{div_type}'")
            return ""

    def extract_text(self):
        """
        Extract text from body tag
        """
        text = ""
        #    if soup.find("div", recursive=False) is None:
        #        # Attempted workaround for TEI
        #        tag_depth = sum(1 for _ in soup.parents) + 1
        #        for div in soup.find_all("div"):
        #            parent_names = [p.name for p in div.parents][::-tag_depth]
        #            # Skip nested divs
        #            if "div" in parent_names:
        #                continue
        #            # Skip divs within edit tags
        #            if "edit" in parent_names:
        #                continue
        #            text += get_div_text(div, message_pfx=filename)
        # Assume core text is in divs
        for div in self.body.find_all("div", recursive=False):
            text += self.get_div_text(div)
        # If no text found, attempt to use p tags for content
        if not text:
            self.print_log("Attempting to use p tags")
            for p_tag in self.body.find_all("p", recursive=False):
                text += self.get_bs4_text(p_tag)

        # Clean resulting text
        final_text = self.clean_text(text)
        # Fail if no text extracted
        assert final_text, f"No text extracted from {self}"
        return final_text


def get_poem_record(poem_fn: pathlib.Path, skip_editions: set[str] = EXCLUDED_EDITIONS):
    poem_parser = TMLParser(poem_fn)
    # Extract Metadata
    poem_metadata = poem_parser.extract_metadata()
    record = {}
    record["poem_id"] = poem_metadata["title"]["title_id"]
    record |= poem_metadata
    # Check if poem should be skipped based on its edition
    if skip_editions:
        edition_id = poem_metadata["title"]["edition_id"]
        if edition_id in skip_editions:
            return None
    # Check if text was extracted succesffully
    extracted_text = poem_parser.extract_text()
    if extracted_text is None:
        # Encountered failure
        return None
    record["text"] = extracted_text
    return record


def get_poem_records(in_dir, poem_ids=None, show_progress=True):
    disable_progress = not show_progress
    desc = "Extracting poems"
    if poem_ids:
        # Gather selected poem records
        poem_set = set(poem_ids) - EXCLUDED_POEMS
        for poem_id in tqdm(
            poem_set, desc=desc, total=len(poem_set), disable=disable_progress
        ):
            poem_dir = in_dir.joinpath(get_poem_subdir(poem_id))
            poem_path = poem_dir.joinpath(f"{poem_id}.tml")
            if not poem_path.is_file():
                print(f"Warning: poem file {poem_path} does not exist")
                continue
            result = get_poem_record(poem_path)
            if result:
                yield get_poem_record(poem_path)
            # else:
            #    print(f"Warning: skipped {poem_path}")
    else:
        # Gather all poem records
        bar_format = "{desc}: {n:,} poems extracted | elapsed: {elapsed}, {rate_fmt}"
        for poem_path in tqdm(
            in_dir.rglob("*.tml"),
            desc=desc,
            bar_format=bar_format,
            dynamic_ncols=True,
            disable=disable_progress,
        ):
            if poem_path.stem in EXCLUDED_POEMS:
                continue
            result = get_poem_record(poem_path)
            if result:
                yield result
            # else:
            #    print(f"Warning: skipped {poem_path}")


def save_poem_records(in_dir, out_fn, poem_ids=None, show_progress=True):
    orjsonl.save(
        out_fn, get_poem_records(in_dir, poem_ids=poem_ids, show_progress=show_progress)
    )


def main():
    parser = argparse.ArgumentParser(description="Build Chadwyck Healey (sub)corpus.")
    # Required
    parser.add_argument(
        "input",
        help="Top-level poem directory",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output",
        help="Filename where the (sub)corpus should be saved",
        type=pathlib.Path,
    )
    # Optional
    parser.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--poem-ids",
        help="Create subcorpus limited to the specified poems (by id)."
        "Can be repeated.",
        nargs="*",
        action="extend",
    )

    args = parser.parse_args()

    # Check that input directory exists
    if not args.input.is_dir():
        print(f"Error: input directory {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
    # Add .jsonl extension to output if no extension is given
    output_fn = args.output
    if not output_fn.suffix:
        output_fn = args.output.with_suffix(".jsonl")
    # Check that output file does not exist
    if output_fn.exists():
        print(f"Error: output file {args.output} already exists", file=sys.stderr)
        sys.exit(1)

    try:
        save_poem_records(
            args.input, output_fn, poem_ids=args.poem_ids, show_progress=args.progress
        )
    except Exception as err:
        # Remove output if it exists
        # args.output.unlink(missing_ok=True)
        # Rethrow error
        raise err


if __name__ == "__main__":
    main()
