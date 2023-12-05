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
        with logwatch(f'booting PPACorpus at {path}'):
            self.path = os.path.abspath(os.path.expanduser(path))
            self.do_clean=clean
            self.path_texts = os.path.join(self.path,texts_dir) if not os.path.isabs(texts_dir) else texts_dir
            self.path_texts_preproc = os.path.join(self.path,texts_preproc_dir) if not os.path.isabs(texts_preproc_dir) else texts_preproc_dir
            self.path_metadata = os.path.join(self.path,metadata_fn) if not os.path.isabs(metadata_fn) else metadata_fn
            self.path_data = os.path.join(self.path, 'data')
            self.path_nlp_db = os.path.join(self.path_data, 'pages_nlp.sqlitedict')
            # self.path_page_db = os.path.join(self.path_data, 'pages.sqlitedict')
            self.path_page_db = os.path.join(self.path_data, 'work_pages.db')
            self.path_page_db_counts = os.path.join(self.path_data, 'work_pages.db.counts')
            self.path_work_ids = os.path.join(self.path_data, 'work_page_ids.json')
            self._topicmodel = None

            # init
            self.textd
            self.page_db

    def __iter__(self): yield from self.iter_texts()

    @cached_property
    def meta(self):
        with logwatch('reading metadata'):
            return pd.read_json(self.path_metadata).fillna('').set_index(self.WORK_ID_FIELD)
    
    @cache
    def get_text(self, work_id):
        return PPAText(work_id, corpus=self)

    @cached_property
    def texts(self):
        return list(self.iter_texts(progress=False))
    @cached_property
    def text_ids(self):
        return list(self.meta.index)
    @cached_property
    def textd(self): return {t.id:t for t in self.texts}
    @property
    def text(self):
        return random.choice(self.texts)
    
    @cached_property
    def num_texts(self):
        return len(self.meta)
    
    @cache
    def ents_db(self, flag='c', autocommit=True):
        return SqliteDict(self.path_nlp_db, flag=flag, tablename='ents', autocommit=autocommit)
    
    @cached_property
    def page_db(self):
        from peewee import SqliteDatabase, Model, CharField, TextField, IntegerField, FloatField

        db = SqliteDatabase(self.path_page_db)

        class BaseModel(Model):
            class Meta:
                database = db

        class Page(BaseModel):
            page_id = CharField()
            page_text = TextField()
            page_num_content_words = IntegerField()
            work_id = CharField()
            cluster = CharField()
            source = CharField()
            year = IntegerField()
            author = CharField()
            title = CharField()
            _random = FloatField()

        db.connect()

        if not db.table_exists(Page): db.create_tables([Page])
        return Page
    
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


    def page_db_count(self, q=None, frac=None, min_doc_len=None,work_ids=None):
        with SqliteDict(self.path_page_db_counts, autocommit=True) as db:
            key=f'frac_{frac}.min_doc_len={min_doc_len}.work_ids_{work_ids}'
            if not key in db:
                with logwatch('counting pages in query'):
                    count = self.page_db.select().where(q).count() if q is not None else self.page_db.select().count()
                db[key]=count
            return db[key]
    
    def iter_pages(self, work_ids=None, clean=None, lim=None, min_doc_len=None, frac=None, max_per_cluster=None, as_dict=True):
        with logwatch('querying page database'):
            Page=self.page_db
            if not frac or frac>1 or frac<=0: frac=None
            i=0
            clustd=Counter()
            work_ids=set(work_ids) if work_ids else None
            
            q=None
            if frac and min_doc_len:
                q = (Page._random<=frac) & (Page.page_num_content_words>=min_doc_len)
            elif frac:
                q = (Page._random<=frac)
            elif min_doc_len:
                q = (Page.page_num_content_words>=min_doc_len)
            
            total = self.page_db_count(q=q,frac=frac,min_doc_len=min_doc_len)
            res = Page.select().where(q) if q is not None else Page.select()
        
        with logwatch('returning page database results'):
            for page_rec in tqdm(res,total=total,desc='Iterating over page search results'):
                d=page_rec.__data__        
                if 'id' in d: del d['id']
                if work_ids and d['work_id'] not in work_ids: continue
                if not max_per_cluster or clustd[d['cluster']]<max_per_cluster:
                    yield PPAPage(d['page_id'], self.textd[d['work_id']], **d) if not as_dict else d
                    if max_per_cluster: clustd[d['cluster']]+=1
                    i+=1
                    if lim and i>=lim: break
                if lim and i>=lim: break

    @cache
    def pages_df(self, **kwargs): 
        return pd.DataFrame(page for page in self.iter_pages(as_dict=True,**kwargs))
    

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

    def gen(self,force=False, num_proc=1):
        num_proc=1  # doesn't work with mp
        done_ids = {rec.work_id for rec in self.page_db.select(self.page_db.work_id).distinct()} if not force else None
        objs = [(id,force) for id in self.text_ids if force or id not in done_ids]

        pmap_run(
            gen_text_pages_mp,
            objs,
            num_proc=num_proc,
            desc='Saving cleaned pages into page db',
            shuffle=True
        )

    def ner_parse_texts(self, lim=25, min_doc_len=25, **kwargs):
        texts=[t for t in self.texts]
        random.shuffle(texts)
        for text in piter(texts,desc='Iterating texts',color='cyan'):
            text.ner_parse(lim=lim, min_doc_len=min_doc_len, **kwargs)
    
    def ner_parse(self, lim=None, **kwargs):
        with self.ents_db(flag='r') as db: done=set(db.keys())
        numdone=Counter(id.split('_')[0] for id in done)
        for page in self.iter_pages(as_dict=False,**kwargs):
            if page.id in done: continue
            if not lim or numdone[page.text.id]<lim:
                page.ner_parse()
                numdone[page.text.id]+=1

    

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
    
    @cache
    def topic_model(self, output_dir=None, model_type=None,ntopic=50, niter=100, min_doc_len=25, frac=1, max_per_cluster=None):
        from .topicmodel import PPATopicModel
        return PPATopicModel(
            output_dir=output_dir,
            corpus=self,
            ntopic=ntopic,
            niter=niter,
            min_doc_len=min_doc_len,
            frac=frac,
            max_per_cluster=max_per_cluster,
            model_type=model_type
        )
    
    @cached_property
    def nlp(self):
        with logwatch('initializing nlp'):
            with logwatch('importing stanza'):
                import stanza
            with logwatch('loading stanza nlp object'):
                try:
                    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
                except Exception:
                    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner',download_method=None)
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
        return f'<b>PPAText({self.id})</b> [{self.author+", " if self.author else ""}<i>{self.title}</i> ({self.year_str})]'

    @cached_property
    def cluster(self): return self.meta.get(CLUSTER_KEY,self.id)
    
    @cached_property
    def source(self): return self.meta.get(SOURCE_KEY,self.id)
    
    @cached_property
    def is_excerpt(self): return self.id != self.source
    
    @cached_property
    def page_ids(self): return self.corpus.page_ids.get(self.id,[])
    
    @cached_property
    def num_pages(self): return len(self.page_ids)
    
    @cached_property
    def meta(self): return dict(self.corpus.meta.loc[self.id])
    
    @cached_property
    def title(self): return self.meta.get('title')
    
    @cached_property
    def author(self): return self.meta.get('author')
    
    @cached_property
    def year_str(self): return str(self.meta.get('pub_date'))[:4]
    
    @cached_property
    def year(self): return pd.to_numeric(self.meta.get('pub_date'),errors='coerce')

    @cached_property
    def path(self): return os.path.join(self.corpus.path_texts, self.meta[self.FILE_ID_KEY]+'.json')
    
    @cached_property
    def path_preproc(self): return os.path.join(self.corpus.path_texts_preproc, self.meta[self.FILE_ID_KEY]+'.json.gz')

    @cached_property
    def pages_json_preproc(self):
        return [PPAPage(d['page_id'], self, **d) for d in self.clean(force=False)]
    
    @cached_property
    def pages_json(self): 
        return [PPAPage(d['page_id'], self, **d) for d in read_json(self.path)]
    
    @cached_property
    def pages(self):
        # if we don't need to clean just return orig json
        if not self.do_clean: return self.pages_json

        # if we already have preproc file just load that
        if os.path.exists(self.path_preproc): return self.pages_json_preproc
        
        # if we only have the db on file use that
        if self.pages_db: return self.pages_db

        # otherwise clean the text and load the result
        return self.pages_json_preproc
    
    
    def iter_pages_db(self, as_dict=True):
        q=(self.corpus.page_db.work_id==self.id)
        # total = self.corpus.page_db_count(q,work_ids=self.id)
        res = self.corpus.page_db.select().where(q)
        for page_rec in tqdm(res,desc='Iterating over page search results'):
            d=page_rec.__data__        
            if 'id' in d: del d['id']
            yield d if as_dict else PPAPage(d['page_id'], self, **d)
    
    @cached_property
    def pages_db(self):
        return list(self.iter_pages_db(as_dict=False))
            
    
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
            return pages_ld
        else:
            return read_json(self.path_preproc)
    
    def iter_page_json(self):
        if os.path.exists(self.path):
            yield from iter_json(self.path)

    @cached_property
    def txt(self, sep='\n\n\n\n'):
        return sep.join(page.txt for page in self.pages)
    
    def gen(self, force=False):
        Page=self.corpus.page_db
        if force or (count:=Page.select().where(Page.work_id==self.id).count()) != self.num_pages:
            if force or count: 
                Page.delete().where(Page.work_id==self.id).execute()

            inp = [
                dict(
                    page_id=page.id,
                    page_text=page.txt,
                    page_num_content_words=page.num_content_words,
                    work_id=page.text.id,
                    cluster = page.text.cluster,
                    source = page.text.cluster,
                    year = page.text.cluster,
                    author = page.text.author,
                    title = page.text.title[:255],
                    _random = random.random()
                )
                for page in self.pages_json_preproc
            ]
            if inp: Page.insert(inp).execute()

    def ner_parse(self, lim=None, min_doc_len=None, **kwargs):
        pages = [page for page in self.pages if not min_doc_len or page.num_content_words>=min_doc_len]
        random.shuffle(pages)
        with self.corpus.ents_db(flag='r') as db: done = {p.id for p in pages if p.id in db}
        undone = [p for p in pages if p.id not in done]
        if not lim or len(done)<lim:
            todo = undone if not lim else undone[:lim-len(done)]
            for page in piter(todo, desc='Iterating pages',color='blue'):
                page.ents
                if 'ents' in page.__dict__: del page.__dict__['ents']



class PPAPage:
    def __init__(self, id, text=None,**_meta):
        self.id = id
        self.text = text if text is not None else PPA().textd[id.split('_')[0]]
        self.corpus = text.corpus
        self._meta = _meta if _meta else text.pages_d.get(self.id)._meta

    
    
    @cached_property
    def meta(self):
        return {
            **self._meta,
            'page_content_words':self.content_words,
            'page_num_tokens':len(self.tokens),
            'page_num_content_words':len(self.content_words),
            'work_id':self.text.id, 
            'cluster':self.text.cluster,
            'source':self.text.source,
            'year':self.text.year,
            'author':self.text.author,
            'title':self.text.title
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
    def content_words(self): 
        return self.get_content_words()
    @cached_property
    def num_content_words(self): 
        res=self._meta.get('page_num_content_words')
        if res is None: res=len(self.content_words)
        return res
    @cached_property
    def stopwords(self): return self.corpus.stopwords

    def get_content_words(self, min_tok_len=4):
        return [tok for tok in self.tokens if len(tok)>=min_tok_len and tok not in self.stopwords]
    

    @cached_property
    def ents(self):
        return self.ner_parse()
    
    def ner_parse(self, db=None):
        if db is None: db = self.corpus.ents_db()
        if self.id in db: return db[self.id]
        doc = self.corpus.nlp(self.txt)
        res = [(ent.text, ent.type) for ent in doc.ents]
        db[self.id] = res
        return res



def cleanup_pages_mp(obj):
    ifn,ofn=obj
    pages_ld=read_json(ifn)
    out=cleanup_pages(pages_ld)
    write_json(out, ofn)


def gen_text_pages_mp(obj):
    work_id,force = obj
    t = PPA().textd[work_id]
    t.gen(force=force)