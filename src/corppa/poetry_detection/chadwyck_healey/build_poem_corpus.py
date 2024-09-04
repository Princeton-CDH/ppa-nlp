"""
Convert the Chadwyck Healey corpus into a single .jsonl file
"""

import argparse
import pathlib
import re
import sys

import bs4
import ftfy
import orjsonl
from tqdm import tqdm

_symbol_table = {
    "&indent;": "\t",
    "&wblank;": "\u2014\u2014",
    "&lblank;": "\u2014",
    "&grwc;": "\u1ff6",
    "&grst;": "\u03c2",
}


def clean_text(text):
    new_text = ftfy.fix_text(text)
    # Complete symbol replacement
    for symbol_str, out_str in _symbol_table.items():
        new_text = new_text.replace(symbol_str, out_str)
    return new_text.rstrip()


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


def extract_tag_data(tag_soup):
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
        tag_info[child_tag.name] = clean_text(child_tag.text)
    return tag_info


def extract_metadata(filename):
    metadata = {}
    with open(filename, encoding="latin1") as reader:
        raw_text = reader.read()
    # This will only parse the "head" block, but won't
    # modify the meta tag
    soup = bs4.BeautifulSoup(raw_text, "lxml-xml")
    for tag in soup.head.find_all(True, recursive=False):
        assert f"{tag.name}_meta" not in metadata
        metadata[f"{tag.name}_meta"] = extract_tag_data(tag)
    return metadata


def get_span_text(span_tag):
    span_text = ""
    # Get span text
    for child in span_tag.children:
        # Children are either NavigableStrings or Tags
        if isinstance(child, bs4.NavigableString):
            span_text += child.text
        elif child.name == "span":
            span_text += get_span_text(child)
        else:
            raise ValueError(f"Unexpected span child tag '{child.name}'")
    # TODO: Set span styling (if this seems at all valuable)
    return span_text


def get_div_text(div_tag):
    if div_tag.name != "div":
        raise ValueError(f"{div_tag} is not a div tag")
    div_type = div_tag["type"]
    # Handle each div type
    if div_type in ["versepara", "stanza"]:
        # Case #1: Group of text (should be separated by whitespace)
        text_block = ""
        # Should contain a group of lines
        for child_div in div_tag.find_all(True, recursive=False):
            assert child_div.name == "div"
            text_block += get_div_text(child_div)
        return text_block + "\n"
    elif div_type in ["line", "firstline", "conclusion", "greek"]:
        # Case #2: Some text
        # There might be weird leading whitespace if a note was skipped
        line = ""
        for child in div_tag.children:
            # Children are either NavigableStrings or Tags
            if isinstance(child, bs4.NavigableString):
                # attempt to remove singleton whitespace
                if child.text == " ":
                    line += ""
                else:
                    line += child.text
            elif child.name == "div":
                line += get_div_text(child)
            elif child.name == "span":
                line += get_span_text(child)
            elif child.name == "edit":
                # Skip editorial tags
                line += ""
            else:
                raise ValueError(f"Unexpected div child tag '{child.name}'")
        return line + "\n"
    elif div_type in [
        "argument",
        "caption",
        "dedication",
        "epigraph",
        "note",
        "preface",
        "signed",
        "trailer",
    ]:
        # Skip these blocks
        return ""
    else:
        print(f"Skipping unexpected div type '{div_type}'", file=sys.stderr)
        return ""


def extract_text(filename):
    with open(filename, encoding="latin1") as reader:
        raw_text = reader.read()
    # Care about body block
    soup = bs4.BeautifulSoup(raw_text, "lxml").body
    text = ""
    if soup.find("div", recursive=False) is None:
        # Attempted workaround for TEI
        tag_depth = sum(1 for _ in soup.parents) + 1
        for div in soup.find_all("div"):
            parent_names = [p.name for p in div.parents][::-tag_depth]
            # Skip nested divs
            if "div" in parent_names:
                continue
            # Skip divs within edit tags
            if "edit" in parent_names:
                continue
            text += get_div_text(div)
    for div in soup.find_all("div", recursive=False):
        # Skip nested div tags
        text += get_div_text(div)
    return clean_text(text)  # Remove trailing whitespace


def get_poem_record(poem_fn):
    poem_metadata = extract_metadata(poem_fn)
    record = {}
    record["poem_id"] = poem_metadata["title_meta"]["title_id"]
    record |= poem_metadata

    try:
        poem_text = extract_text(poem_fn)
    except (ValueError, NotImplementedError) as err:
        raise type(err)(f"{poem_fn.stem}: {err}")

    record["text"] = poem_text
    return record


def get_poem_records(in_dir, poem_ids=None, show_progress=True):
    disable_progress = not show_progress
    desc = "Extracting poems"
    if poem_ids:
        # Gather selected poem records
        poem_set = set(poem_ids)
        for poem_id in tqdm(
            poem_set, desc=desc, total=len(poem_set), disable=disable_progress
        ):
            poem_dir = in_dir.joinpath(get_poem_subdir(poem_id))
            poem_path = poem_dir.joinpath(f"{poem_id}.tml")
            if not poem_path.is_file():
                print(f"Warning: poem file {poem_path} does not exist")
                continue
            yield get_poem_record(poem_path)
    else:
        # Gather all poem records
        bar_format = "{desc}: {n:,} poems extracted | elapsed: {elapsed}, {rate_fmt}"
        for poem_path in tqdm(
            in_dir.rglob("*.tml"),
            desc=desc,
            bar_format=bar_format,
            disable=disable_progress,
        ):
            yield get_poem_record(poem_path)


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
        help="FIlename where the (sub)corpus should be saved",
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
        "--poems",
        help="Create subcorpus limited to the specified poems (by id)."
        "Can be repeated/",
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
    if output_fn.is_file():
        print(f"Error: output file {args.output} already exists", file=sys.stderr)
        sys.exit(1)

    try:
        save_poem_records(
            args.input, args.output, poem_ids=args.poems, show_progress=args.progress
        )
    except Exception as err:
        # Remove output if it exists
        args.output.unlink(missing_ok=True)
        # Rethrow error
        raise err


if __name__ == "__main__":
    main()
