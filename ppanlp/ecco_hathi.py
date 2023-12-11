from .imports import *
from .cleanup import cleanup_pages


def get_page_cache(fn=PATH_ECCO_PAGES_CACHE):
    return SqliteDict(fn, autocommit=True, encode=encode_cache, decode=decode_cache)

def get_pages(text_id, page_id=None):
    try:
        with get_page_cache() as cache:
            return cache[text_id][page_id] if page_id else cache[text_id]
    except KeyError:
        return {}
    
# @cache
# def get_ecco_metadata(fns = PATHS_METADATA):
#     l = [pd.read_csv(fn) for fn in fns]
#     df = pd.concat(l) if l else pd.DataFrame()
#     assert len(df) and 'productlink' in set(df.columns)
#     df['source_id'] = df['productlink'].apply(lambda x: x.split('GALE%7C')[1])
#     return df.set_index('source_id')

def get_sourcepage_id(source_id, pages_orig):
    x,y=source_id,pages_orig
    return (f'{x}_{str(y).split("-")[0]}' if y else x).replace('/','|')

@cache
def get_ppa_metadata():
    odf=pd.read_csv(PATH_PPA_METADATA).fillna('')
    odf['sourcepage_id']=[
        get_sourcepage_id(x,y)
        for x,y in zip(odf.source_id, odf.pages_orig)
    ]
    return odf.set_index('sourcepage_id')

# @cache
# def get_ppa_metadata_d():
#     return {source_id:dict(row) for source_id,row in get_ppa_metadata().iterrows()}




# @cache
# def get_ecco_pages():
#     # Function to get pages for text, using sqlitedict cache made from ECCO page XMLs

#     with get_page_cache() as cache:
#         def get_pages_row(row):
#             textid=row.psmid
#             textpages = cache.get(textid)
#             if not textpages: return []
            
#             opages = []
#             pagerange=row.pages_digital
#             if not pagerange: 
#                 opages = list(textpages.values())
#             else:
#                 for page_num in intspan(pagerange):
#                     page_id = f'{page_num:04}0'
#                     page = textpages.get(page_id)
#                     if page:
#                         opages.append(page)
            
#             for d in opages:
#                 d['source_id']=row.source_id

#             return opages
#         df_excerpts = get_ppa_metadata()
#         df_meta = get_ecco_metadata()
#         df_merged = df_excerpts.merge(df_meta, on='source_id').reset_index()
#         all_pages = [page for i,row in tqdm(df_merged.iterrows(), total=len(df_merged)) for page in get_pages_row(row)]
#         pages_df = pd.DataFrame(all_pages, dtype=str).set_index('source_id')
#         return pages_df
    
# @cache
# def get_all_ecco_data():
#     df_excerpts = get_ppa_metadata()
#     df_meta = get_ecco_metadata()
#     df_pages = get_ecco_pages()
#     return df_pages.merge(df_excerpts, on='source_id').merge(df_meta, on='source_id')

# def iter_pages_ecco():
#     pages_df=get_all_ecco_data()
#     for i,row in pages_df.iterrows():
#         recd=dict(row)
#         recd['source_id']=recd['productlink'].split('GALE%7C')[-1]
#         yield recd

def iter_pages_hathi():
    with jsonlines.open(PATH_HATHI_PAGE_JSONL) as reader:
        for rec in tqdm(reader,total=1498802):
            recd=dict(rec)
            recd['pageid']=recd.pop('page_id')
            recd['text']=recd.pop('content')
            yield recd#{**recd, **get_ppa_metadata_d().get(recd['source_id'], {})}

def iterate_over_all_ppa_pages():
    yield from iter_pages_ecco()
    yield from iter_pages_hathi()



def get_ppa_page_cache(fn=PATH_PPA_PAGES_CACHE):
    return SqliteDict(fn, autocommit=True, encode=encode_cache, decode=decode_cache)


@cache
def get_txtcorpus_metadata(): return pd.read_csv(PATH_TEXT_CORPUS_METADATA)






### ECCO


def _iter_all_ecco_files():
    yield from os.walk(PATH_CORPUS_RAW_ECCO1)
    yield from os.walk(PATH_CORPUS_RAW_ECCO2)
def iter_all_ecco_files():
    for root,dirs,fns in _iter_all_ecco_files():
        for x in fns:
            if x.endswith('.xml'):
                yield os.path.join(root,x)

def get_all_ecco_paths():
    # Get all XML files
    all_xml_paths = list(tqdm(iter_all_ecco_files()))
    def path2id(path): return os.path.basename(path).split('_')[0]
    text_ids = sorted(list({path2id(fn) for fn in all_xml_paths}))
    textid2paths = {k:{} for k in text_ids}
    for path in all_xml_paths:
        assert 'PageText' in path or 'DocMetadata' in path
        key='meta' if 'DocMetadata' in path else 'text'
        textid2paths[path2id(path)][key]=path
    return textid2paths

@cache
def get_ecco_metadata():
    dfecco = pd.concat([
        pd.read_csv(os.path.join(PATH_REPO_DATA,'data.all_xml_metadata.ECCO1.csv'),dtype=str),
        pd.read_csv(os.path.join(PATH_REPO_DATA,'data.all_xml_metadata.ECCO2.csv'),dtype=str),
    ])
    dfecco['source_id']=dfecco.productlink.apply(lambda x: x.split('GALE%7C')[-1])
    pathd = get_all_ecco_paths()
    dfecco['paths']=dfecco.psmid.apply(lambda x: pathd[x])
    return dfecco



def get_ecco_pages_from_paths(paths):
    path_meta=paths.get('meta')
    path_text=paths.get('text')

    with open(path_meta) as fmeta, open(path_text) as ftxt:
        dom_meta = bs4.BeautifulSoup(fmeta.read())
        dom_text = bs4.BeautifulSoup(ftxt.read())

    pages_meta = dom_meta('page')
    attrs_meta = ['pageid','assetid','ocrlanguage','ocr','imagelink']
    ld_meta = [
        {**{ak:page.find(ak).text for ak in attrs_meta}, **page.attrs}
        for page in pages_meta
    ]

    pages_text = dom_text('page')
    ld_text = [{'pageid':page.get('id'), 'text':page.find('ocrtext').text, **{k:v for k,v in page.attrs.items() if k!='id'}} 
               for page in pages_text]

    # join both on page id
    odf=pd.DataFrame(ld_meta).merge(
        pd.DataFrame(ld_text), 
        on='pageid',
        how='outer'
    )
    # quant cols?
    odf['ocr'] = pd.to_numeric(odf['ocr'],errors='coerce')
    odf['page_len_char']=odf['text'].apply(lambda x: len(x.strip()))
    odf['page_len_word']=odf['text'].apply(lambda x: len(x.strip().split()))
    return odf.to_dict('records')

def save_ecco_files_from_row(row):
    paths = row.paths
    sourcepage = row.sourcepage_id
    for d in get_ecco_pages_from_paths(paths):
        pageid1 = d['pageid']
        pageid=str(int(pageid1))
        opath = os.path.join(PATH_TEXT_CORPUS_TEXTS_PAGES, sourcepage)
        os.makedirs(opath, exist_ok=True)
        ofn=os.path.join(opath,pageid+'.txt')
        with open(ofn,'w') as of: of.write(d.pop('text'))
        d={
            'id':f'{sourcepage}/{pageid}',
            'sourcepage_id':sourcepage,
            **{k:v for k,v in d.items() if k not in {'id','sourcepage_id'}}
        }
        yield d

def save_plaintext_ecco_files():
    dfppa = get_ppa_metadata().reset_index()
    dfppa_ecco = dfppa[dfppa.source=='Gale']
    dfecco = get_ecco_metadata()
    dfppa_ecco_ecco = dfppa_ecco.merge(dfecco, on='source_id').fillna('')

    old=[]
    for i,row in tqdm(dfppa_ecco_ecco.iterrows(), total=len(dfppa_ecco_ecco)):
        for od in save_ecco_files_from_row(row):
            old.append(od)
    odf=pd.DataFrame(old)
    odf.to_csv(os.path.join(PATH_REPO_DATA, 'data.all_ppa_ecco_pagedata.csv'), index=False)
    return odf



#### HATHI

def save_plaintext_hathi_files():
    iterr_all = iter_pages_hathi()
    os.makedirs(PATH_TEXT_CORPUS_TEXTS_PAGES, exist_ok=True)
    ld=[]
    for d in tqdm(iterr_all,total=1498802):
        sourcepage=get_sourcepage_id(d['source_id'], d['pages_orig'])
        pageid=str(int(d['pageid']))
        d['id']=id=f'{sourcepage}/{pageid}'
        d['sourcepage_id']=sourcepage
        txt=d.pop('text')
        opath = os.path.join(PATH_TEXT_CORPUS_TEXTS_PAGES, sourcepage)
        os.makedirs(opath, exist_ok=True)
        ofn=os.path.join(opath,pageid+'.txt')
        with open(ofn,'w') as of: of.write(txt)
        # ld.append({k:v for k,v in d.items() if k in common_keys})
        ld.append(d)
    df=pd.DataFrame(ld)
    df.to_csv(os.path.join(PATH_REPO_DATA, 'data.all_ppa_hathi_pagedata.csv'), index=False)
    return df



### mini corpus

pagetyperenames=dict(
    bodyPage='BODY_PAGE',
    frontmatter='FRONT_MATTER',
    backmatter='BACK_MATER',
    index='INDEX',
    TOC='TABLE_OF_CONTENTS',
    titlePage='TITLE_PAGE'
)

def gen_plaintext_mini_corpus(ecco=True, hathi=True):
    if ecco:
        dfppa = get_ppa_metadata().reset_index()
        dfppa_ecco = dfppa[dfppa.source=='Gale']
        dfecco = get_ecco_metadata()
        dfppa_ecco_ecco = dfppa_ecco.merge(dfecco, on='source_id').fillna('')    
        for i,row in tqdm(dfppa_ecco_ecco.iterrows(), total=len(dfppa_ecco_ecco)):
            if row.pages_digital: 
                pagespan = set(intspan(row.pages_digital))
            else:
                pagespan = None
            
            for d in get_ecco_pages_from_paths(row.paths):
                page_i = int(d['pageid'])
                if d['pageid'].endswith('0'): 
                    page_i = page_i//10
                    if pagespan and page_i not in pagespan:
                        # print(d['pageid'],page_i,pagespan,row.pages_digital)
                        continue
                
                page_types=[pagetyperenames[d['type']]]
                if d['firstpage']=='yes': page_types.append('FIRST_PAGE')
                od={
                    'sourcepage_id':row.sourcepage_id,
                    'page_id':d['pageid'],
                    'page_i':page_i,
                    'page_text':d['text'],
                    'page_types':page_types
                }
                yield od

    if hathi:
        for d in iter_pages_hathi():
            sourcepage=get_sourcepage_id(d['source_id'], d['pages_orig'])
            if d['pages_digital']: 
                pagespan = set(intspan(row.pages_digital))
            else:
                pagespan = None
            
            page_i = int(d['pageid'])
            if pagespan and page_i not in pagespan:
                continue
                
            od={
                'sourcepage_id':sourcepage,
                'page_id':d['pageid'],
                'page_i':page_i,
                'page_text':d['text'],
                'page_types':d['tags']
            }
            yield od

def save_plaintext_mini_corpus(ofn=PATH_TEXT_CORPUS_MINI):
    with jsonlines.open(ofn,'w') as of:
        for d in gen_plaintext_mini_corpus():
            of.write(d)

def iter_plaintext_mini_corpus(fn=PATH_TEXT_CORPUS_MINI):
    with jsonlines.open(fn) as f:
        for x in tqdm(f,total=1893144):
            yield x

iter_ppa_pages = iter_plaintext_mini_corpus


def get_ppa_page_cache(fn=PATH_PPA_PAGES_CACHE):
    return SqliteDict(fn, autocommit=True, encode=encode_cache, decode=decode_cache)



def save_plaintext_mini_corpus_jsons(clean=True):
    ## metadata
    df=get_ppa_metadata()
    os.makedirs(PATH_JSON_CORPUS_TEXTS, exist_ok=True)
    df.to_csv(PATH_JSON_CORPUS_METADATA)

    ## data
    last_id=None
    out=[]

    def save_current(out, last_id):
        ofn=os.path.join(PATH_JSON_CORPUS_TEXTS, last_id+'.json.gz')
        if clean: out = cleanup_pages(out)
        out_b = orjson.dumps(out,option=orjson.OPT_INDENT_2)
        with gzip.open(ofn,'w') as of: 
            of.write(out_b)

    for d in iter_ppa_pages():
        id=d.pop('sourcepage_id')
        if last_id and id!=last_id and out:
            save_current(out, last_id)
            out = []
        last_id=id
        out.append(d)
    if last_id and out:
        save_current(out, last_id)





class PPACorpus:
    path_texts = PATH_JSON_CORPUS_TEXTS
    path_metadata = PATH_JSON_CORPUS_METADATA

    @cached_property
    def meta(self):
        df_metadata=pd.read_csv(self.path_metadata)
        # add filename
        df_metadata['filename'] = df_metadata['sourcepage_id'].apply(lambda id: os.path.join(self.path_texts,id+'.json'))
        df_metadata['filename_exists'] = df_metadata['filename'].progress_apply(os.path.exists)
        return df_metadata.set_index('sourcepage_id')
    
    @cache
    def text(self, sourcepage_id):
        row=self.meta.loc[sourcepage_id]
        return PPAText(**dict(row))
        
    
class PPAText:
    def __init__(self, **attrs):
        for k,v in attrs.items(): setattr(self,k,v)

    @cached_property
    def pages(self):
        if not self.filename_exists: return []
        with open(self.filename) as f:
            return orjson.loads(f.read())