from .imports import *

def iter_json(fn):
    if os.path.exists(fn):
        if fn.endswith('.gz'):
            with gzip.open(fn, 'rt', encoding='UTF-8') as zipfile:
                yield from json.load(zipfile)
        else:
            with open(fn, 'r', encoding='UTF-8') as f:
                yield from json.load(f)

def write_json(obj, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    if fn.endswith('.gz'):
        with gzip.open(fn, 'wt', encoding='UTF-8') as zipfile:
            json.dump(obj, zipfile)
    else:
        with open(fn, 'w', encoding='UTF-8') as of:
            json.dump(obj, of)