from .imports import *


def encode_cache(x): return zlib.compress(orjson.dumps(x))
def decode_cache(x): return orjson.loads(zlib.decompress(x))
def get_page_cache(fn=PATH_ECCO_PAGES_CACHE):
    return SqliteDict(fn, autocommit=True, encode=encode_cache, decode=decode_cache)

def get_pages(text_id, page_id=None):
    try:
        with get_page_cache() as cache:
            return cache[text_id][page_id] if page_id else cache[text_id]
    except KeyError:
        return {}
    
@cache
def get_ecco_metadata(fns = PATHS_METADATA):
    l = [pd.read_csv(fn) for fn in fns]
    df = pd.concat(l) if l else pd.DataFrame()
    assert len(df) and 'productlink' in set(df.columns)
    df['source_id'] = df['productlink'].apply(lambda x: x.split('GALE%7C')[1])
    return df.set_index('source_id')

@cache
def get_ecco_excerpts():
    return pd.read_csv(PATH_ECCO_EXCERPTS).set_index('source_id')

@cache
def get_ecco_pages():
    # Function to get pages for text, using sqlitedict cache made from ECCO page XMLs

    with get_page_cache() as cache:
        def get_pages_row(row):
            textid=row.psmid
            textpages = cache.get(textid)
            if not textpages: return []
            
            opages = []
            pagerange=row.pages_digital
            if not pagerange: 
                opages = list(textpages.values())
            else:
                for page_num in intspan(pagerange):
                    page_id = f'{page_num:04}0'
                    page = textpages.get(page_id)
                    if page:
                        opages.append(page)
            
            for d in opages:
                d['source_id']=row.source_id

            return opages
        df_excerpts = get_ecco_excerpts()
        df_meta = get_ecco_metadata()
        df_merged = df_excerpts.merge(df_meta, on='source_id').reset_index()
        all_pages = [page for i,row in tqdm(df_merged.iterrows(), total=len(df_merged)) for page in get_pages_row(row)]
        pages_df = pd.DataFrame(all_pages, dtype=str).set_index('source_id')
        return pages_df
    
@cache
def get_all_ecco_data_for_excerpts():
    df_excerpts = get_ecco_excerpts()
    df_meta = get_ecco_metadata()
    df_pages = get_ecco_pages()
    return df_pages.merge(df_excerpts, on='source_id').merge(df_meta, on='source_id')
