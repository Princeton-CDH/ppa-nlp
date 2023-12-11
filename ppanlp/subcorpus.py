from .imports import *

class PPASubcorpus(PPACorpus):
    def __init__(self, path_metadata):
        path_metadata = os.path.abspath(path_metadata)
        path = os.path.splitext(path_metadata)[0]
        os.makedirs(path,exist_ok=True)
        new_path_metadata = os.path.join(path, 'metadata'+os.path.splitext(path_metadata)[-1])
        if not os.path.exists(new_path_metadata):
            os.symlink(path_metadata,new_path_metadata)
        super().__init__(
            path=path,
            metadata_fn = path_metadata,
        )
        self.install()

    @cached_property
    def page_ranges(self):
        return {
            id:set(intspan(prange)) if prange else prange
            for id,prange in zip(self.meta.index, self.meta.page_nums)
        }
    
    @cached_property
    def meta_ppa(self):
        return self.meta.join(PPA().meta,rsuffix='_ppa')

    
    def install(self,clear=True,force=False, batchsize=1000):
        if (
            not force 
            and os.path.exists(self.path_pages_jsonl)
            and self.page_db_count 
            and get_num_lines_json(self.path_pages_jsonl, progress=False) == self.page_db_count
        ):
            return

        self.cleardb()
        if 'page_db_count' in self.__dict__: del self.__dict__['page_db_count']
        with logwatch(f'generating pages jsonl and db') as lw, gzip.open(self.path_pages_jsonl,'wt') as of:
            batch=[]
            def savebatch():
                with logwatch('inserting into db', level='TRACE'):
                    self.page_db.insert(batch).execute()
            for page in self.iter_pages_from_ppa():
                of.write(json.dumps(dict(page.meta))+'\n')
                batch.append(page.db_input)
                if len(batch)>=batchsize:
                    savebatch()
                    batch=[]
            if batch: savebatch()

    def iter_pages_from_ppa(self, as_dict=False):
        for page in PPA().iter_pages(work_ids=tuple(self.text_ids)):
            if page.text.id in self.page_ranges:
                pagerange = self.page_ranges[page.text.id]
                if not pagerange or page.num in pagerange:
                    yield page if not as_dict else dict(page.meta)

    