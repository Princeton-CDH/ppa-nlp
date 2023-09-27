from .imports import *


def encode_cache(x): return zlib.compress(orjson.dumps(x))
def decode_cache(x): return orjson.loads(zlib.decompress(x))
def get_page_cache(fn=path_ecco_pages_cache):
    return SqliteDict(fn, autocommit=True, encode=encode_cache, decode=decode_cache)

def get_pages(text_id, page_id=None):
    try:
        with get_page_cache() as cache:
            return cache[text_id][page_id] if page_id else cache[text_id]
    except KeyError:
        return {}
    
@cache
def get_metadata():
    files_root_eccoII = os.listdir(path_root_eccoII)
    excel_file_paths = [os.path.join(path_root_eccoII,fn) for fn in files_root_eccoII if fn.endswith('.xlsx')]
    return pd.concat(
        pd.read_excel(path).assign(subcorpus=os.path.splitext(os.path.basename(path))[0])
        for path in tqdm(excel_file_paths)
    )