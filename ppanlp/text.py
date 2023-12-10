from .imports import *

class PPAText:
    FILE_ID_KEY='work_id'

    def __init__(self, id, corpus=None,clean=None):
        self.id=id
        self.corpus=corpus if corpus is not None else PPA()
        self.do_clean=self.corpus.do_clean if clean==None else clean

    def __getitem__(self, page_id):
        return self.pages_d.get(page_id)

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
    def num_pages_db(self): return self.corpus.page_db_counts.get(self.id,0)
    
    @cached_property
    def meta(self): return dict(self.corpus.meta.loc[self.id])
    
    @cached_property
    def title(self): return self.meta.get('title')
    
    @cached_property
    def author(self): return self.meta.get('author')
    
    @cached_property
    def year_str(self): return str(self.meta.get('pub_date'))[:4]
    
    @cached_property
    def year(self): 
        try:
            return int(self.year_str)
        except:
            return 0

    @cached_property
    def path(self): return os.path.join(self.corpus.path_texts, clean_filename(self.id+'.jsonl'))
    
    @cached_property
    def path_preproc(self):
        return os.path.join(
            self.corpus.path_texts_preproc,
            clean_filename(self.id+'.jsonl.gz')
        ) 

    @cached_property
    def pages_preproc(self): 
        return list(self.iter_pages_preproc())
    @cached_property
    def pages_orig(self): 
        return list(self.iter_pages_orig())
    
    def iter_pages_preproc(self, force_clean=False, as_dict=False):
        from .page import PPAPage
        self.clean(force=force_clean)
        try:
            for d in iter_json(self.path_preproc):
                yield PPAPage(d['page_id'], self, **d)
        except Exception as e:
            if not force_clean:
                # could be that cleaned file was saved improperly
                yield from self.iter_pages_preproc(force_clean=True)
            else:
                raise e

    
    def iter_pages_orig(self, as_dict=False):        
        from .page import PPAPage
        def iter_dicts():
            if os.path.exists(self.path):
                yield from self.iter_page_json()
        
        if as_dict:
            yield from iter_dicts()
        else:
            for d in iter_dicts():
                yield PPAPage(d['page_id'], self, **d)
        
    @cached_property
    def pages(self):
        # if we don't need to clean just return orig json
        if not self.do_clean: return self.pages_json

        # if we already have preproc file just load that
        if os.path.exists(self.path_preproc): 
            return self.pages_preproc
        
        # if we only have the db on file use that
        if self.pages_db: return self.pages_db

        # otherwise clean the text and load the result
        return self.pages_preproc
    
    
    def iter_pages_db(self, as_dict=True, progress=True):
        from .page import PPAPage
        q=(self.corpus.page_db.work_id==self.id)
        # total = self.corpus.page_db_count(q,work_ids=self.id)
        res = self.corpus.page_db.select().where(q)
        for page_rec in tqdm(res,desc='Iterating over page search results',position=0,disable=not progress):
            d=page_rec.__data__        
            if 'id' in d: del d['id']
            yield d if as_dict else PPAPage(d['page_id'], self, **d)
    
    @cached_property
    def pages_db(self):
        return list(self.iter_pages_db(as_dict=False, progress=False))
            
    
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
    
    @cached_property
    def is_in_db(self):
        return self.num_pages == self.num_pages_db

    # def gen_db(self, force=False, delete_existing = True):
    #     if force or self.num_pages_db != self.num_pages:
    #         if (force or self.num_pages_db) and delete_existing: 
    #             self.page_db.delete().where(self.page_db.work_id==self.id).execute()
    #         inp = [page.db_input for page in self.iter_pages_preproc()]
    #         if inp: 
    #             self.page_db.insert(inp).execute()

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
