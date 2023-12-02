from .imports import *
from .cleanup import cleanup_pages

PPA_OBJ=None
PATH_CORPUS=PATH_PPA_CORPUS

def set_corpus_path(path=None,**kwargs):
    global PPA_OBJ,PATH_CORPUS
    PATH_CORPUS=os.path.abspath(os.path.expanduser(path)) if path else PATH_PPA_CORPUS
    PPA_OBJ = PPACorpus(PATH_CORPUS,**kwargs)

def PPA(path=None, **kwargs):
    if PPA_OBJ == None or path or kwargs: set_corpus_path(path, **kwargs)
    return PPA_OBJ


class PPACorpus:
    WORK_ID_FIELD = 'id'

    def __init__(self, path:str, clean=False, texts_dir='texts', metadata_fn='metadata.json', texts_preproc_dir='texts_preproc'):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.do_clean=clean
        self.path_texts = os.path.join(self.path,texts_dir) if not os.path.isabs(texts_dir) else texts_dir
        self.path_texts_preproc = os.path.join(self.path,texts_preproc_dir) if not os.path.isabs(texts_preproc_dir) else texts_preproc_dir
        self.path_metadata = os.path.join(self.path,metadata_fn) if not os.path.isabs(metadata_fn) else metadata_fn
        self.path_data = os.path.join(self.path, 'data')
        self.path_nlp_db = os.path.join(self.path_data, 'nlp.db')
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
    @property
    def text(self):
        return random.choice(self.texts)
    


    def iter_texts(self, work_ids=None):
        if work_ids is None: work_ids=self.meta.index
        pdesc='Iteration over texts in PPA'
        pbar = tqdm(total=len(work_ids), position=0, desc=pdesc)
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
                for page in text.iter_pages(clean=clean):
                    if not min_doc_len or page.num_content_words>=min_doc_len:
                        if frac==1 or random.random()<=frac:
                            if not max_per_cluster or clustd[page.text.cluster]<max_per_cluster:
                                yield page
                                clustd[page.text.cluster]+=1
                                i+=1
                                if lim and i>=lim: break
            if lim and i>=lim: break

    def clean_texts(self, work_ids=None, num_proc=None):
        work_ids = list(self.meta.index) if work_ids == None else work_ids
        num_proc=(mp.cpu_count()//2)-1 if not num_proc else num_proc
        num_proc=num_proc if num_proc>0 else 1
        pool = mp.Pool(num_proc)

        random.shuffle(work_ids)
        objs = work_ids
        res = list(tqdm(pool.imap_unordered(cleanup_mp,objs), total=len(objs), position=0, desc=f'Cleaning PPA pages (script from Wouter Haverals) [{num_proc}x]'))
        return res
    

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

    def __iter__(self): yield from self.iter_pages()

    @cached_property
    def id(self): return self.meta.get('work_id')

    @cached_property
    def cluster(self): return self.meta.get('cluster_id_s',self.id)
    @cached_property
    def source(self): return self.meta.get('source_id',self.id)
    @cached_property
    def is_excerpt(self): return self.id != self.source

    @cached_property
    def meta(self):
        return dict(self.corpus.meta.loc[self.id])

    @cached_property
    def path(self):
        return os.path.join(self.corpus.path_texts, self.meta[self.FILE_ID_KEY]+'.json')
    
    @cached_property
    def path_preproc(self):
        return os.path.join(self.corpus.path_texts_preproc, self.meta[self.FILE_ID_KEY]+'.json.gz')

    @cached_property
    def pages_df(self):
        return pd.DataFrame([p.meta for p in self.iter_pages()]).set_index('page_id')

    @cached_property
    def pages(self): 
        return list(self.iter_pages())
    @cached_property
    def pages_orig(self): 
        return list(self.iter_pages_orig())
    @property
    def page(self):
        return random.choice(self.pages)

    def clean(self,remove_headers=True,force=False):
        self.do_clean=True
        for k in ['pages','pages_df','txt']:
            if k in self.__dict__: del self.__dict__[k]

        if force or not os.path.exists(self.path_preproc):
            pages_ld = list(self.iter_pages_orig(as_dict=True))
            new_pages = cleanup_pages(pages_ld, remove_headers=remove_headers)

            os.makedirs(os.path.dirname(self.path_preproc),exist_ok=True)
            with open(self.path_preproc,'w') as of:
                json.dump(new_pages, of, indent=2)
        return self
    
    def iter_pages_orig(self, as_dict=False):
        if os.path.exists(self.path):
            for d in iter_json(self.path):
                yield PPAPage(self, **d) if not as_dict else d
    
    def iter_pages_preproc(self, as_dict=False):
        self.clean()
        if os.path.exists(self.path_preproc):
            for d in iter_json(self.path):
                yield PPAPage(self, **d) if not as_dict else d

    def iter_pages(self, clean=None):
        clean = self.do_clean if clean==None else clean
        yield from self.iter_pages_preproc() if clean else self.iter_pages_orig()

    @cached_property
    def txt(self, sep='\n\n\n\n'):
        return sep.join(page.txt for page in self.pages)

def cleanup_mp(work_id):
    t = PPA().get_text(work_id)
    t.clean()
    return t.path_preproc


class PPAPage:
    def __init__(self, text, **page_d):
        self.text = text
        self.corpus = text.corpus
        self.d = page_d

    @cached_property
    def meta(self):
        return {'work_id':self.text.id, **self.d}
    
    @cached_property
    def id(self):
        return self.meta.get('page_id')

    @cached_property
    def txt(self):
        return self.meta.get('page_text')
    
    @cached_property
    def tokens(self):
        tokens=self.meta.get('page_tokens')
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
        with SqliteDict(self.corpus.path_nlp_db, tablename='ents', autocommit=True) as db:
            if self.id in db: 
                return db[self.id]

            doc = self.corpus.nlp(self.txt)
            res = [(ent.text, ent.type) for ent in doc.ents]
            db[self.id] = res
            return res
    