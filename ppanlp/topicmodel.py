from .imports import *
from .corpus import PPA
import tomotopy as tp

class PPATopicModel:
    def __init__(
            self, 
            output_dir=None,
            path_docs=None,
            corpus=None,
            ntopic=50, 
            niter=100, 
            clean=None, 
            min_doc_len=25,
            frac=1,
            frac_text=1,
            max_per_cluster=None,
            ):

        self.corpus=corpus if corpus!=None else PPA()
        self.ntopic = ntopic
        self.niter = niter
        self.clean = self.corpus.do_clean if clean == None else clean
        self.min_doc_len = min_doc_len
        self.frac=frac
        self.frac_text=frac_text
        self.max_per_cluster=max_per_cluster
        self._mdl = None
        self.id2index = {}
        self.index2id = {}

        self.path_topicmodel = os.path.join(self.corpus.path_data,'topicmodels') 
        self.path = output_dir if output_dir and os.path.isabs(output_dir) else os.path.join(
            self.path_topicmodel,
            'models',
            (
                output_dir if output_dir else (
                    f'data.tomotopy.model.ntopic_{self.ntopic}.niter_{self.niter}.min_doc_len_{self.min_doc_len}.frac_{self.frac}.max_per_cluster_{self.max_per_cluster}'
                )
            )
        )
        # self.path_corpus = path_docs if path_docs else os.path.join(
        #     self.path_topicmodel, 
        #     'corpora', 
        #     f'data.minicorpus.min_doc_len_{self.min_doc_len}.frac_text_{self.frac_text}.frac_{self.frac}.max_per_cluster_{self.max_per_cluster}.jsonl.gz'
        # )

        self.path_model = os.path.join(self.path, 'model.bin')
        self.path_index = os.path.join(self.path, 'index.json')
        self.path_params = os.path.join(self.path, 'params.json')
        self.path_ldavis = os.path.join(self.path, 'ldavis')

    # def prepare_docs(self, force=False, lim=None):
    #     if not force and os.path.exists(self.path_corpus):
    #         return
        
    #     def iterr():
    #         for page in self.corpus.iter_pages(lim=lim,min_doc_len=self.min_doc_len,frac=self.frac,max_per_cluster=self.max_per_cluster):
    #             yield dict(
    #                 work_id=page.text.id,
    #                 page_id=page.id,
    #                 page_words=page.content_words
    #             )
        
    #     write_jsonl(
    #         iterr(),
    #         self.path_corpus
    #     )
        
    
    # def iter_docs(self, lim=None, progress=True):
    #     if not os.path.exists(self.path_corpus): self.prepare_docs(lim=lim)
    #     iterr=iter_jsonl(self.path_corpus)
    #     if progress:
    #         total=get_num_lines_json(self.path_corpus)
    #         iterr=tqdm(iterr, total=total, desc='Iterating documents', position=0)
    #     yield from iterr
    def iter_docs(self, lim=None):
        yield from self.corpus.iter_pages(
            lim=lim,
            min_doc_len=self.min_doc_len,
            frac=self.frac,
            max_per_cluster=self.max_per_cluster
        )



    def model(self, output_dir=None,force=False, lim=None):
        
        # get filename
        fdir=self.path
        os.makedirs(fdir, exist_ok=True)
        fn=self.path_model
        fnindex=self.path_index
        fnparams=self.path_params

        # save?
        if force or not os.path.exists(fn) or not os.path.exists(fnindex):
            mdl = self.mdl = tp.LDAModel(k=self.ntopic)
            docd=self.id2index={}
            for page in self.iter_docs(lim=lim):
                tokens = page['page_content_words']
                docd[page['page_id']] = mdl.add_doc(tokens)

            def getdesc():
                return f'Training model (ndocs={len(docd)}, log-likelihood = {mdl.ll_per_word:.4})'
            
            pbar=tqdm(list(range(0, self.niter, 1)),desc=getdesc(),position=0)
            
            for i in pbar:
                pbar.set_description(getdesc())
                mdl.train(1)
            mdl.save(fn)
            print('Saved:',fn)
            with open(fnindex,'wb') as of:
                of.write(orjson.dumps(docd,option=orjson.OPT_INDENT_2))
            mdl.summary()
            print('Saved:',fnindex)

            params=dict(
                ntopic=self.ntopic,
                niter=self.niter,
                clean=self.clean,
                min_doc_len=self.min_doc_len,
                frac=self.frac,
                lim=lim
            )
            with open(fnparams,'wb') as of:
                of.write(orjson.dumps(params,option=orjson.OPT_INDENT_2))

        else:
            print('Loading:',fn)
            self.mdl = tp.LDAModel.load(fn)
            with open(fnindex,'rb') as f:
                self.id2index=orjson.loads(f.read())
        self.index2id={v:k for k,v in self.id2index.items()}
    
    
    @cached_property
    def topic_term_dists(self): 
        return np.stack([self.mdl.get_topic_word_dist(k) for k in range(self.mdl.k)])

    @cached_property
    def doc_topic_dists(self):
        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in self.mdl.docs])
        doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
        return doc_topic_dists

    @cached_property
    def mdl(self):
        if self._mdl == None:
            self.model()
        return self._mdl

    @cached_property
    def doc_lengths(self):
        return np.array([len(doc.words) for doc in self.mdl.docs])

    @cached_property
    def vocab(self): return list(self.mdl.used_vocabs)
    @cached_property
    def term_frequency(self): return self.mdl.used_vocab_freq

    def save_pyldavis(self, output_dir=None):
        import pyLDAvis
        prepared_data = pyLDAvis.prepare(
            self.topic_term_dists, 
            self.doc_topic_dists, 
            self.doc_lengths, 
            self.vocab, 
            self.term_frequency,
            start_index=0, # tomotopy starts topic ids with 0, pyLDAvis with 1
            sort_topics=False # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
        )
        
        output_dir = self.path_ldavis if not output_dir else output_dir
        os.makedirs(output_dir, exist_ok=True)
        fn=os.path.join(output_dir, 'index.html')
        print('Saving:',fn)
        pyLDAvis.save_html(prepared_data, fn)
        return fn
    

    @cached_property
    def doc_topic_dists_df(self):
        page_ids,values = zip(*[(self.index2id[i],x) for i,x in enumerate(self.doc_topic_dists) if i in self.index2id])
        dftopicdist = pd.DataFrame(values)
        dftopicdist['page_id'] = page_ids
        dftopicdist['work_id']=[i.split('_')[0] for i in page_ids]
        dftopicdist['cluster']=[self.id2cluster.get(work_id,work_id) for work_id in dftopicdist.work_id]
        return dftopicdist.set_index(['cluster','work_id','page_id'])
    
    @cached_property
    def id2cluster(self):
        return dict(zip(self.corpus.meta.index, self.corpus.meta[CLUSTER_KEY]))

    @cached_property
    def meta(self):
        df = self.doc_topic_dists_df        
        df_avgs=df.groupby('work_id').mean(numeric_only=True)
        return self.corpus.meta.merge(df_avgs, on='work_id').set_index('work_id')