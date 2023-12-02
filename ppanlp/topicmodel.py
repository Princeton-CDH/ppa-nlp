from .imports import *
from .corpus import PPA

class PPATopicModel:
    def __init__(
            self, 
            output_dir=None,
            corpus=None,
            ntopic=50, 
            niter=100, 
            clean=None, 
            min_doc_len=25,
            frac=1,
            frac_text=1,
            ):

        self.corpus=corpus if corpus!=None else PPA()
        self.ntopic = ntopic
        self.niter = niter
        self.clean = self.corpus.do_clean if clean == None else clean
        self.min_doc_len = min_doc_len
        self.frac=frac
        self.frac_text=frac_text
        self._mdl = None
        self.id2index = {}
        self.index2id = {}

        self.path_topicmodel = os.path.join(self.corpus.path_data,'topicmodels') 
        self.path = os.path.join(
            self.path_topicmodel,
            'models',
            (
                output_dir if output_dir else (
                    f'data.tomotopy.model.ntopic_{self.ntopic}.niter_{self.niter}.clean_{self.clean}.min_doc_len_{self.min_doc_len}.frac_text_{self.frac_text}.frac_{self.frac}'
                )
            )
        )
        self.path_corpus = os.path.join(
            self.path_topicmodel, 
            'corpora', 
            f'data.minicorpus.clean_{self.clean}.min_doc_len_{self.min_doc_len}.frac_text_{self.frac_text}.frac_{self.frac}.jsonl.gz'
        )
        self.path_model = os.path.join(self.path, 'model.bin')
        self.path_index = os.path.join(self.path, 'index.json')
        self.path_params = os.path.join(self.path, 'params.json')
        self.path_ldavis = os.path.join(self.path, 'ldavis')

    def prepare_corpus(self, force=False, lim=None):
        def iterr():
            for page in self.corpus.iter_pages(clean=self.clean,lim=lim,min_doc_len=self.min_doc_len,frac=self.frac,frac_text=self.frac_text):
                yield dict(
                    work_id=page.text.id,
                    page_id=page.id,
                    page_words=page.content_words
                )
        
        write_jsonl(
            iterr(),
            self.path_corpus
        )
    

    def model(self, output_dir=None,force=False, lim=None):
        import tomotopy as tp
        
        # get filename
        fdir=self.path
        os.makedirs(fdir, exist_ok=True)
        fn=self.path_model
        fnindex=self.path_index
        fnparams=self.path_params

        # save?
        if force or not os.path.exists(fn) or not os.path.exists(fnindex):
            mdl = self.mdl = tp.LDAModel(k=50)
            docd=self.id2index={}
            for page in self.corpus.iter_pages(clean=self.clean,lim=lim):
                tokens = page.content_words
                if not self.min_doc_len or len(tokens)>=self.min_doc_len:
                    if self.frac==1 or random.random()<=self.frac:
                        docd[page.id] = mdl.add_doc(tokens)

            def getdesc():
                return f'Training model (ndocs={len(docd)}, log-likelihood = {mdl.ll_per_word:.4})'
            
            pbar=tqdm(list(range(0, self.niter, 10)),desc=getdesc(),position=0)
            
            for i in pbar:
                pbar.set_description(getdesc())
                mdl.train(10)
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
            self.mdl = tp.LDAModel.load(fn)
            with open(fnindex,'rb') as f:
                self.id2index=orjson.loads(f.read())
            print('Loaded:',fn)
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