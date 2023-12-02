from .imports import *

def iter_json(fn):
    if 'jsonl' in fn:
        yield from iter_jsonl(fn)
    else:
        yield from read_json(fn)

def iter_jsonl(fn):
    if os.path.exists(fn):
        yield from orjsonl.stream(fn)
                
def write_jsonl(obj, fn):
    ensure_dir(fn)
    orjsonl.save(fn, obj)

def read_jsonl(fn):
    return list(iter_jsonl(fn))


def read_json(fn):
    if os.path.exists(fn):
        if 'jsonl' in fn:
            return read_jsonl(fn)
        if fn.endswith('.gz'):
            with gzip.open(fn, 'r') as zipfile:
                return orjson.loads(zipfile.read())
        else:
            with open(fn, 'rb') as f:
                return orjson.loads(f.read())
    return []

def write_json(obj, fn):    
    if 'jsonl' in fn: return write_jsonl(obj, fn)
    ensure_dir(fn)
    
    if fn.endswith('.gz'):
        with gzip.open(fn, 'w') as zipfile:
            zipfile.write(orjson.dumps(obj))
    else:
        with open(fn, 'wb') as of:
            of.write(orjson.dumps(obj,option=orjson.OPT_INDENT_2))


def tokenize_agnostic(txt):
    return re.findall(r"[\w']+|[.,!?; -—–'\n]", txt)

def untokenize_agnostic(l):
    return ''.join(l)

def ensure_dir(fn):
    dirname=os.path.dirname(fn)
    if dirname: os.makedirs(dirname, exist_ok=True)
    
def get_num_lines_json(fn, progress=True):
    if not os.path.exists(fn): return
    nl=sum(1 for _ in tqdm(iter_json(fn), desc=f'Counting lines in {fn[fn.index(".json"):] if ".json" in fn else fn}',disable=not progress,position=0))
    return nl