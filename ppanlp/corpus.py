from .imports import *
from .cleanup import cleanup_pages

PPA_OBJ=None
PATH_CORPUS=PATH_PPA_CORPUS

def set_corpus_path(path=None,**kwargs):
    global PPA_OBJ,PATH_CORPUS
    PATH_CORPUS=os.path.abspath(os.path.expanduser(path.strip())) if path else PATH_PPA_CORPUS
    PPA_OBJ = PPACorpus(PATH_CORPUS,**kwargs)

def PPA(path=None, **kwargs):
    if PPA_OBJ == None or path or kwargs: set_corpus_path(path, **kwargs)
    return PPA_OBJ


class PPACorpus:
    WORK_ID_FIELD = 'id'

    def __init__(self, path:str, clean=True, texts_dir='texts', metadata_fn='metadata.json', texts_preproc_dir='texts_preproc'):
        path=path.strip()
        self.path = os.path.abspath(os.path.expanduser(path))
        self.do_clean=clean
        self.path_texts = os.path.join(self.path,texts_dir) if not os.path.isabs(texts_dir) else texts_dir
        self.path_texts_preproc = os.path.join(self.path,texts_preproc_dir) if not os.path.isabs(texts_preproc_dir) else texts_preproc_dir
        self.path_metadata = os.path.join(self.path,metadata_fn) if not os.path.isabs(metadata_fn) else metadata_fn
        self.path_data = os.path.join(self.path, 'data')
        self.path_nlp_db = os.path.join(self.path_data, 'pages_nlp.sqlitedict')
        # self.path_page_db = os.path.join(self.path_data, 'pages.sqlitedict')
        self.path_page_db = os.path.join(self.path_data, 'work_pages.sqlitedict')
        self.path_work_ids = os.path.join(self.path_data, 'work_page_ids.json')
        self._topicmodel = None

    def __iter__(self): yield from self.iter_texts()

    @cached_property
    def meta(self):
        return pd.read_json(self.path_metadata).fillna('').set_index(self.WORK_ID_FIELD)
    
    @cache
    def get_text(self, work_id):
        return PPAText(work_id, corpus=self)

    @cached_property
    def texts(self):
        return list(self.iter_texts())
    @cached_property
    def text_ids(self):
        return list(self.meta.index)
    @cached_property
    def textd(self): return {t.id:t for t in self.iter_texts()}
    @property
    def text(self):
        return random.choice(self.texts)
    
    @cached_property
    def num_texts(self):
        return len(self.meta)
    
    def ents_db(self, flag='c', autocommit=True):
        return SqliteDict(self.path_nlp_db, flag=flag, tablename='ents', autocommit=autocommit)
    
    @cache
    def page_db(self, flag='c', autocommit=True):
        return CompressedSqliteDict(self.path_page_db, flag=flag, autocommit=autocommit)
    
    @cached_property
    def page_ids(self):
        self.index(force=False)
        return read_json(self.path_work_ids)
    

    def iter_texts(self, work_ids=None,progress=True,desc=None):
        if work_ids is None: work_ids=self.meta.index
        pdesc='Iteration over texts in PPA' if not desc else desc
        pbar = tqdm(total=len(work_ids), position=0, desc=pdesc,disable=not progress)
        for work_id in work_ids:
            # pbar.set_description(f'{pdesc}: {work_id}')
            yield self.get_text(work_id)
            pbar.update()

    def iter_pages(self, work_ids=None, clean=None, lim=None, min_doc_len=None, frac=1, frac_text=1, max_per_cluster=None):
        if not frac or frac>1 or frac<=0: frac=1
        if not frac_text or frac_text>1 or frac_text<=0: frac_text=1

        i=0
        clustd=Counter()
        for text in self.iter_texts(work_ids=work_ids):
            if frac_text==1 or random.random()<=frac_text:
                for page in text:
                    if not min_doc_len or page.num_content_words>=min_doc_len:
                        if frac==1 or random.random()<=frac:
                            if not max_per_cluster or clustd[page.text.cluster]<max_per_cluster:
                                yield page
                                clustd[page.text.cluster]+=1
                                i+=1
                                if lim and i>=lim: break
            if lim and i>=lim: break

    def index(self, force=False):
        if force or not os.path.exists(self.path_work_ids):
            wdb = {}
            for text in self.iter_texts(desc='Indexing page ids by work'):
                ids=[page['page_id'] for page in text.iter_page_json()]
                wdb[text.id] = ids
            write_json(wdb, self.path_work_ids)

    def clean(self, num_proc=1, force=False):
        objs=[
            (t.path,t.path_preproc) 
            for t in self.iter_texts(desc='Gathering texts needing cleaning') 
            if force or not t.is_cleaned
        ]
        if objs:
            pmap_run(
                cleanup_pages_mp,
                objs,
                num_proc=num_proc,
                shuffle=True,
                desc='Cleaning up texts and storing in texts_preproc'
            )

    

    @cached_property
    def stopwords(self):
        from nltk.corpus import stopwords as stops
        try:
            stopwords = set(stops.words('english'))
        except Exception:
            import nltk
            nltk.download('stopwords')  
            stopwords = set(stops.words('english'))
        return stopwords
    
    def topic_model(self, output_dir=None, ntopic=50, force=False, niter=100, clean=None, lim=None, min_doc_len=25, frac=1, frac_text=1, max_per_cluster=None):
        from .topicmodel import PPATopicModel

        if not force and self._topicmodel!=None:
            return self._topicmodel
        
        self._topicmodel = PPATopicModel(
            output_dir=output_dir,
            corpus=self,
            ntopic=ntopic,
            niter=niter,
            clean=clean,
            min_doc_len=min_doc_len,
            frac=frac,
            frac_text=frac_text,
            max_per_cluster=None
        )
        return self._topicmodel
    
    @cached_property
    def nlp(self):
        import stanza
        nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
        return nlp
    
        











class PPAText:
    FILE_ID_KEY='work_id'

    def __init__(self, id, corpus=None,clean=None):
        self.id=id
        self.corpus=corpus if corpus is not None else PPA()
        self.do_clean=self.corpus.do_clean if clean==None else clean

    def __iter__(self): yield from self.pages
    def __repr__(self):
        return f'''PPAText({self.id})'''
    def _repr_html_(self):
        return f'<b>PPAText({self.id})</b> [{self.author+", " if self.author else ""}<i>{self.title}</i> ({str(self.year)[:4]})]'

    @cached_property
    def cluster(self): return self.meta.get('cluster_id_s',self.id)
    @cached_property
    def source(self): return self.meta.get('source_id',self.id)
    @cached_property
    def is_excerpt(self): return self.id != self.source

    @cached_property
    def page_ids(self):
        return self.corpus.page_ids.get(self.id,[])

    @cached_property
    def meta(self):
        return dict(self.corpus.meta.loc[self.id])
    @cached_property
    def title(self): return self.meta.get('title')
    @cached_property
    def author(self): return self.meta.get('author')
    @cached_property
    def year(self): return self.meta.get('pub_date')

    @cached_property
    def path(self):
        return os.path.join(self.corpus.path_texts, self.meta[self.FILE_ID_KEY]+'.json')
    
    @cached_property
    def path_preproc(self):
        return os.path.join(self.corpus.path_texts_preproc, self.meta[self.FILE_ID_KEY]+'.json.gz')

    @cached_property
    def pages_clean(self):
        self.clean(force=False)
        return [PPAPage(d['page_id'], self, **d) for d in iter_json(self.path_preproc)]
    
    @cached_property
    def pages_orig(self): 
        return [PPAPage(d['page_id'], self, **d) for d in iter_json(self.path)]
    
    @cached_property
    def pages(self):
        return self.pages_clean if self.do_clean else self.pages
    
    @cached_property
    def pages_d(self):
        return {page.id:page for page in self.pages}

    @cached_property
    def pages_df(self):
        return pd.DataFrame([p.meta for p in self.pages]).set_index('page_id')
    @property
    def page(self):
        return random.choice(self.pages)
    @property
    def is_cleaned(self):
        return os.path.exists(self.path_preproc)

    def clean(self,remove_headers=True,force=False):
        if force or not self.is_cleaned:
            pages_ld = read_json(self.path)
            pages_ld = cleanup_pages(pages_ld, remove_headers=remove_headers)
            write_json(pages_ld, self.path_preproc)
    
    def iter_page_json(self):
        if os.path.exists(self.path):
            yield from iter_json(self.path)

    @cached_property
    def txt(self, sep='\n\n\n\n'):
        return sep.join(page.txt for page in self.pages)



class PPAPage:
    def __init__(self, id, text=None,**_meta):
        self.id = id
        self.text = text if text is not None else PPA().textd[id.split('_')[0]]
        self.corpus = text.corpus
        self._meta = _meta

    
    
    @cached_property
    def meta(self):
        return {
            'work_id':self.text.id, 
            **self._meta,
            'page_num_tokens':len(self.tokens),
            'page_num_content_words':len(self.content_words)
        }
    
    @cached_property
    def txt(self):
        return self._meta.get('page_text')
    
    @cached_property
    def tokens(self):
        tokens=self._meta.get('page_tokens')
        if not tokens: tokens=tokenize_agnostic(self.txt)
        tokens = [x.strip().lower() for x in tokens if x.strip() and x.strip()[0].isalpha()]
        return tokens

    @cached_property
    def content_words(self): return self.get_content_words()
    @cached_property
    def num_content_words(self): return len(self.content_words)
    @cached_property
    def stopwords(self): return self.corpus.stopwords

    def get_content_words(self, min_tok_len=4):
        return [tok for tok in self.tokens if len(tok)>=min_tok_len and tok not in self.stopwords]
    

    @cached_property
    def ents(self):
        ensure_dir(self.corpus.path_nlp_db)
        with self.corpus.ents_db as db:
            if self.id in db: 
                return db[self.id]

            doc = self.corpus.nlp(self.txt)
            res = [(ent.text, ent.type) for ent in doc.ents]
            db[self.id] = res
            return res
    


def cleanup_pages_mp(obj):
    ifn,ofn=obj
    pages_ld=read_json(ifn)
    out=cleanup_pages(pages_ld)
    write_json(out, ofn)