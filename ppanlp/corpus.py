from .imports import *

PPA_OBJ=None
PATH_CORPUS=PATH_PPA_CORPUS

def get_work_id(page_id): return '.'.join(page_id.split('.')[:-1])
def is_page_id(page_id): return page_id.split('.')[-1].isdigit()


def set_corpus_path(path=None,**kwargs):
    global PPA_OBJ,PATH_CORPUS
    PATH_CORPUS=os.path.abspath(os.path.expanduser(path.strip())) if path else PATH_PPA_CORPUS
    PPA_OBJ = PPACorpus(PATH_CORPUS,**kwargs)

def PPA(path=None, **kwargs):
    if PPA_OBJ == None or path or kwargs: set_corpus_path(path, **kwargs)
    return PPA_OBJ


class PPACorpus:
    WORK_ID_FIELD = 'group_id_s'
    NUM_LINES_JSONL = 2160441
    QUANT_COLS = ['pub_year']


    def __init__(self, path:str, clean=True, texts_dir='texts', metadata_fn='metadata.json', pages_fn='pages.jsonl.gz',texts_preproc_dir='texts_preproc'):
        path=path.strip()
        with logwatch(f'booting PPACorpus at {truncfn(path)}'):
            self.do_clean=clean
            self.path = os.path.abspath(os.path.expanduser(path))
            self.path_pages_jsonl = os.path.join(self.path,pages_fn) if not os.path.isabs(pages_fn) else pages_fn
            self.path_texts = os.path.join(self.path,texts_dir) if not os.path.isabs(texts_dir) else texts_dir
            self.path_texts_preproc = os.path.join(self.path,texts_preproc_dir) if not os.path.isabs(texts_preproc_dir) else texts_preproc_dir
            self.path_metadata = os.path.join(self.path,metadata_fn) if not os.path.isabs(metadata_fn) else metadata_fn
            self.path_data = os.path.join(self.path, 'data')
            self.path_nlp_db = os.path.join(self.path_data, 'pages_nlp.sqlitedict')
            # self.path_page_db = os.path.join(self.path_data, 'pages.sqlitedict')
            self.path_page_db = os.path.join(self.path_data, 'pages.sqlite')
            self.path_page_db_counts = os.path.join(self.path_data, 'page_query_cache.sqlitedict')
            self.path_work_ids = os.path.join(self.path_data, 'work_page_ids.json')
            self._topicmodels = {}

            # init
            self.textd
            self.page_db

    def __iter__(self): yield from self.iter_texts()

    @cached_property
    def paths(self):
        return {k:v for k,v in ppa.__dict__.items() if k.startswith('path')}

    def __getitem__(self, work_or_page_id):
        if is_page_id(work_or_page_id):
            work_id = get_work_id(work_or_page_id)
            text = self.get_text(work_id)
            return text[work_or_page_id]
        else:
            return self.get_text(work_or_page_id)

    @cached_property
    def meta(self):
        with logwatch('reading metadata'):
            df=read_df(self.path_metadata).fillna('')
            for col in self.QUANT_COLS:
                df[col] = pd.to_numeric(df[col],errors='coerce')
            return df.set_index('work_id')
    
    
    @cache
    def get_text(self, work_id):
        from .text import PPAText
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
        return CompressedSqliteDict(self.path_nlp_db, flag=flag, tablename='ents', autocommit=autocommit)
    
    @cached_property
    def _page_db_conn(self):
        from peewee import SqliteDatabase
        ensure_dir(self.path_page_db)
        return SqliteDatabase(self.path_page_db)

    @cached_property
    def page_db(self):
        from peewee import Model, CharField, TextField, IntegerField, FloatField

        db=self._page_db_conn

        class BaseModel(Model):
            class Meta:
                database = db

        class Page(BaseModel):
            page_id = CharField(primary_key=True)
            page_text = TextField()
            page_num = IntegerField()
            page_num_orig = CharField()
            page_num_cluster = IntegerField(null=True)
            page_num_content_words = IntegerField(null=True)
            page_num_content_words_work = IntegerField(null=True)
            page_num_content_words_cluster = IntegerField(null=True)
            page_ocr_accuracy = FloatField(null=True)
            work_id = CharField()
            cluster = CharField()
            source = CharField()
            year = IntegerField()
            author = CharField()
            title = CharField()
            _random = FloatField()

            class Meta:
                indexes = (
                    # create a unique on from/to/date
                    (('page_num_content_words', 'page_num_content_words_work', 'page_num_content_words_cluster','_random'), False),
                    (('page_num_content_words', 'page_num_content_words_cluster','_random'), False),
                    (('page_num_content_words','_random'), False),
                    (('page_num_content_words','_random'), False),
                    (('work_id',), False),
                )

        db.connect()

        if not db.table_exists(Page): db.create_tables([Page])
        return Page
    
    @cached_property
    def page_ids(self):
        # self.index(force=False)
        res = read_json(self.path_work_ids)
        return {} if not res else res
    
    @cached_property
    def num_pages(self):
        return sum(len(v) for k,v in self.page_ids.items())

    def iter_texts(self, work_ids=None,progress=True,desc=None):
        pdesc='iterating over texts in PPA' if not desc else desc
        with logwatch(pdesc) as lw:
            if work_ids is None: work_ids=self.meta.index
            iterr = lw.iter_progress(work_ids,disable=not progress)
            yield from (self.get_text(work_id) for work_id in iterr)

    @cached_property
    def page_db_query_cache(self):
        return SqliteDict(self.path_page_db_counts, autocommit=True)
    
    @cached_property
    def page_db_count(self):
        return self.page_db.select().count()
    @cached_property
    def page_db_counts(self):
        with logwatch('gathering page counts by work in db'):
            return Counter(
                res.work_id
                for res in self.page_db.select(self.page_db.work_id)
            )

    def get_page_db_query(self, frac=None,frac_min=0,frac_max=1,min_doc_len=None, work_ids=None, max_work_len=None, max_cluster_len=None):
        # build query
        if frac is not None and 0<frac<1: frac_max=frac_min+frac
        if not min_doc_len or min_doc_len<1: min_doc_len=0
        frac_min,frac_max=round(frac_min,4),round(frac_max,4)

        

        # if work_ids:
        #     q = (self.page_db.work_id.in_(set(work_ids)))
        # elif max_cluster_len and max_work_len:
        #     q = (
        #         (frac_min<=self.page_db._random<frac_max)
        #         & (self.page_db.page_num_content_words>=min_doc_len)
        #         & (self.page_db.page_num_content_words_cluster<=max_cluster_len)
        #         & (self.page_db.page_num_content_words_work<=max_work_len)
        #     )
        # elif max_cluster_len:
        #     q = (
        #         (frac_min<=self.page_db._random<frac_max)
        #         & (self.page_db.page_num_content_words>=min_doc_len)
        #         & (self.page_db.page_num_content_words_cluster<=max_cluster_len)
        #     )
        # else:
        #     q = (
        #         (frac_min<=self.page_db._random<frac_max)
        #         & (self.page_db.page_num_content_words>=min_doc_len)
        #     )


        ql=[
            '(frac_min<=self.page_db._random<frac_max)',
            '(self.page_db.page_num_content_words>=min_doc_len)',
        ]
        if work_ids:
            ql.append('(self.page_db.work_id.in_(set(work_ids)))')
        
        if max_cluster_len:
            ql.append('(self.page_db.page_num_content_words_cluster<=max_cluster_len)')

        if max_work_len:
            ql.append('(self.page_db.page_num_content_words_work<=max_work_len)')

        qstr = ' & '.join(ql)
        q=eval(qstr)
        # find total
        keyd={
            'work_ids':work_ids,
            'min_doc_len':min_doc_len,
            'max_cluster_len':max_cluster_len,
            'max_work_len':max_work_len,
            'frac_min':frac_min, 
            'frac_max':frac_max
        }
        keyj = json.dumps(keyd)
        key = hashstr(keyj)
        with self.page_db_query_cache as pdqc:
            if key in pdqc:
                total = pdqc[key]
            else:
                with logwatch(f'counting rows for query ({keyj})'):
                    total = self.page_db.select().where(q).count()
                pdqc[key]=total
        # query
        res = self.page_db.select().where(q)
        return (res,total,keyj)
    
    
    def iter_pages(self, use_db=True, as_dict=False, preproc=True, **query_kwargs):
        if (use_db or query_kwargs) and os.path.exists(self.path_page_db):
            yield from self.iter_pages_db(as_dict=as_dict, **query_kwargs)
        else:
            if query_kwargs: 
                logger.warning(f'no filtering applied to query-less json iteration: {query_kwargs}')
            if preproc:
                yield from self.iter_pages_text_jsons(preproc=True,as_dict=as_dict)
            elif os.path.exists(self.path_pages_jsonl):
                yield from self.iter_pages_jsonl(as_dict=as_dict)
            else:
                raise Exception('no page source')

            
    def iter_pages_text_jsons(self, preproc=True, **kwargs):
        with logwatch('iterating pages by individual text json files'):
            for text in self.iter_texts():
                yield from text.iter_pages_orig(**kwargs) if not preproc else text.iter_pages_preproc(**kwargs)
        
    def iter_pages_db(self, work_ids=None, clean=None, lim=None, min_doc_len=None, frac=None, frac_min=0, max_per_cluster=None, max_cluster_len=None,max_work_len=None, as_dict=False):
        from .page import PPAPage
        with logwatch('iterating pages by page database'):
            i=0
            work_ids=set(work_ids) if work_ids else None
            res,total,qkey = self.get_page_db_query(frac=frac,min_doc_len=min_doc_len,frac_min=frac_min,max_cluster_len=max_cluster_len,max_work_len=max_work_len,work_ids=work_ids)
            
            with logwatch(f'iterating page database results ({qkey})') as lw:
                for page_rec in lw.iter_progress(res,total=total,desc='iterating over page search results',position=0):
                    d=page_rec.__data__        
                    if 'id' in d: del d['id']
                    yield PPAPage(d['page_id'], self.textd[d['work_id']], **d) if not as_dict else d
                    i+=1
                    if lim and i>=lim: break

    @cache
    def pages_df(self, **kwargs): 
        return pd.DataFrame(page for page in self.iter_pages(as_dict=True,**kwargs)).set_index('page_id')
    
    def iter_pages_jsonl(self, as_dict=False, desc=None,progress=True):
        from .page import PPAPage
        fn=self.path_pages_jsonl
        iterr=iter_json(fn)
        
        def iter_func():
            for d in iterr:
                if not d['work_id'] in self.textd:
                    raise Exception('work not found: '+str(d))
                yield d if as_dict else PPAPage(
                    d['page_id'],
                    self.textd.get(d['work_id']),
                    **d
                )
        if progress:
            desc='iterating pages by corpus jsonl file' if desc is None else desc
            with logwatch(desc) as lw:
                yield from lw.iter_progress(
                    iter_func(),
                    total=self.num_pages,
                    disable=not progress
                )
        else:
            yield from iter_func()
            

    def index(self, force=False):
        with logwatch('indexing corpus, storing page ids per work'):
            if force or not os.path.exists(self.path_work_ids):
                wdb=defaultdict(list)
                for d in self.iter_pages_jsonl(as_dict=True):
                    wdb[d['work_id']].append(d['page_id'])
                write_json(wdb, self.path_work_ids)
        
        # refresh
        for k in ['page_ids', 'num_pages']:
            if k in self.__dict__:
                del self.__dict__[k]
    
    def install(self, num_proc=None, force=False, clear=False):
        with logwatch('installing corpus: indexing, preprocessing, saving to sqlite'):
            self.index()
            self.preproc(num_proc=num_proc, force=force)
            self.gen_db(force=force, startover=clear)

    def preproc(self, num_proc=None, force=False, shuffle=True, lim=None, max_queue=None):
        with logwatch(f'preprocessing jsonl files'):
            last_work_id=None
            last_pages=[]
            resl=[]
            work_ids_done=set()
            wdb=defaultdict(set)
            successful = []
            errors = []
            os.makedirs(self.path_texts_preproc,exist_ok=True)
            numdone=0
            tries=0

            with mp.get_context(CONTEXT).Pool(num_proc) as pool:
                if num_proc is None: 
                    num_proc=mp.cpu_count() // 2 - 1
                    if num_proc<1: 
                        num_proc=1

                if max_queue is None: 
                    max_queue = 100


                def getdesc():
                    return f'{max_queue} texts in queue, {numdone} finished; {format_timespan(tries)} since last completion'


                with logwatch(f'saving jsonl files to {self.path_texts_preproc} [{num_proc}x]') as lw:
                    iterr = self.iter_pages_jsonl(
                        as_dict=True, 
                        desc=f"preprocessing and saving pages with a multiprocessing pool of {num_proc} CPUs", 
                        progress=False,
                    )
                    iterr = lw.iter_progress(
                        iterr,
                        total=self.num_pages,
                        desc=getdesc()
                    )
                    for d in iterr:
                        work_id=d.get('work_id')
                        wdb[work_id].add(d['page_id'])
                        if last_pages and work_id!=last_work_id:
                            assert last_work_id not in work_ids_done, \
                                'we assume that original jsonl file is sorted by work id'
                            work_ids_done.add(last_work_id)
                            ofn=os.path.join(
                                self.path_texts_preproc,
                                clean_filename(last_work_id+'.jsonl.gz')
                            )
                            if force or not os.path.exists(ofn):
                                res = pool.apply_async(
                                    save_cleanup_pages, 
                                    args=(
                                        last_pages,
                                        ofn
                                    )
                                )
                                resl.append(res)
                                tries=0
                                while len(resl)>=max_queue:
                                    tries+=1
                                    for i,res in enumerate([x for x in resl]):
                                        try:
                                            if res.successful():
                                                successful.append(res)
                                                resl.pop(i)
                                                numdone+=1
                                                tries=0
                                            else:
                                                errors.append(res)
                                        except ValueError:
                                            pass
                                    time.sleep(1)
                                    lw.set_progress_desc(getdesc())
                            
                            last_pages = []
                        last_work_id=work_id
                        last_pages.append(d)
                    
                    for res in lw.iter_progress(resl,desc=f'preprocessing remaining texts [{num_proc}x]',position=0): 
                        res.get()
        

    @cached_property
    def lemmatizer(self):
        from nltk.stem import WordNetLemmatizer
        return WordNetLemmatizer()
    
    @cache
    def lemmatize(self, word):
        import nltk
        try:
            return self.lemmatizer.lemmatize(word)
        except LookupError:
            nltk.download('wordnet')
            return self.lemmatize(word)

    def cleardb(self):
        conn=self.__dict__.get('_page_db_conn')
        db=self.__dict__.get('page_db')
        if conn is not None: 
            conn.close()
            self.__dict__.pop('_page_db_conn')
        if db is not None:
            self.__dict__.pop('page_db')
        if os.path.exists(self.path_page_db): 
            os.unlink(self.path_page_db)

    def gen_db(self,force=False,startover=False, batchsize=1000):
        if startover:
            force=True
            self.cleardb()
            
        with logwatch(f'generating page database at {self.path_page_db}'):
            clustcount=Counter()
            clustwordcount=Counter()
            workwordcount=Counter()
            batch = []
            for text in self.iter_texts(desc='saving preprocessed pages to database'):
                if (force or not text.is_in_db) and text.is_cleaned:
                    self.page_db.delete().where(self.page_db.work_id==text.id).execute()

                    for page in text.iter_pages_preproc():
                        clustcount[page.text.cluster]+=1
                        clustwordcount[page.text.cluster]+=page.num_content_words
                        workwordcount[page.text.id]+=page.num_content_words
                        inpd = {
                            **page.db_input, 
                            'page_num_cluster':clustcount[page.text.cluster], 
                            'page_num_content_words_work':workwordcount[page.text.id],
                            'page_num_content_words_cluster':clustwordcount[page.text.cluster],
                        }
                        batch.append(inpd)

                        if len(batch)>=batchsize:
                            self.page_db.insert(batch).execute()
                            batch = []
            if batch:
                self.page_db.insert(batch).execute()

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


    def ner_model(self, **kwargs):
        from .ner import NERModel
        return NERModel(self, **kwargs)

    

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
    
    def topic_model(self, model_type=None, **query_kwargs):
        from .topicmodel import PPATopicModel
        mdl = PPATopicModel(
            model_type=model_type,
            corpus=self,
            **query_kwargs
        )
        return mdl
    
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
    
        














def cleanup_pages_mp(obj):
    ifn,ofn=obj
    pages_ld=read_json(ifn)
    out=cleanup_pages(pages_ld)
    write_json(out, ofn)


def gen_text_pages_mp(obj):
    work_id,force = obj
    t = PPA().textd[work_id]
    t.gen_pagedb(force=force)

def save_cleanup_pages(pages_ld, save_to):
    pages_ld=cleanup_pages(pages_ld)
    write_json(pages_ld, save_to)

def save_orig_pages(pages_ld, save_to):
    pages_ld=cleanup_pages(pages_ld)
    write_json(pages_ld, save_to)

def preproc_json(obj):
    ifn,ofn,force = obj
    if force or not os.path.exists(ofn):
        pages_ld = read_json(ifn)
        pages_ld=cleanup_pages(pages_ld)
        write_json(pages_ld, ofn)









