from .imports import *
from .corpus import PPA
import tomotopy as tp

class BaseTopicModel:
    model_type='topicmodel_type'

    attrs_set_after_modeling = [
        'doc_df',
        'work_df',
        'path_ldavis',
        'path_model',
        'path_index',
        'path_params',
        'path_topicdf',
        'path_hclust',
        'path_docdf',
        'topic_term_dists',
        'doc_topic_dists',
        'doc_lengths',
        'vocab',
        'term_frequency',
        'doc_topic_dists_df',
        'id2cluster',
        'meta',
        'topic_names',
        'num_topics'
    ]

    def __init__(
            self, 
            corpus=None,
            **query_kwargs
            ):

        self.corpus=corpus if corpus!=None else PPA()
        self.query_kwargs = query_kwargs
        self._mdl = None
        self.id2index = {}
        self.index2id = {}
        self._id_docs,self._ids,self._docs = None,None,None
        self.path_topicmodels = os.path.join(self.corpus.path_data,'topicmodels') 
        self.paths = {}
        
        
    def get_paths(self, output_dir=None, ntopic=None, niter=None,lemmatize=False):
        if output_dir:
            path = os.path.abspath(output_dir)
        else:
            pieces = {
                **self.query_kwargs,
                'ntopic':ntopic,
                'niter':niter,
                'lemmatize':lemmatize,
            }
            def get_fn(pieces): return '.'.join(f'{k}_{v}' for k,v in pieces.items() if v)
            path=os.path.join(self.path_topicmodels, self.model_type, get_fn(pieces))
        
        outd = dict(
            path = path,
            path_model = os.path.join(path, 'model.bin'),
            path_index = os.path.join(path, 'index.json'),
            path_params = os.path.join(path, 'params.json'),
            path_docdf = os.path.join(path, 'documents.xlsx'),
            path_topicdf = os.path.join(path, 'topics.xlsx'),
            path_ldavis = os.path.join(path, 'ldavis'),
            path_hclust = os.path.join(path, 'hclust.html'),
        )
        self.paths = outd
        return outd
    
    def __getattr__(self, k):
        if k.startswith('path'): return self.paths.get(k)
        return None

    def iter_pages(self, lim=None):
        yield from self.corpus.iter_pages(
            lim=lim,
            as_dict=False,
            **self.query_kwargs
        )

    def iter_docs(self, lim=None, lemmatize=False, orig_txt=False, as_str=True):
        yield from (
            (
                page.id,
                page.get_tokens(lemmatize=lemmatize,orig=orig_txt,as_str=as_str)
            )
            for page in self.iter_pages(lim=lim)
        )


    def model(self, **kwargs):
        return # IMPLEMENT
    
    @cached_property
    def mdl(self):
        if self._mdl == None: self.model()
        return self._mdl

    def init_docs(self, lim=None, force=False, lemmatize=False):
        if force or self._id_docs is None:
            for bk in ['id_docs','docs','ids','doc2id','id2doc']:
                if bk in self.__dict__:
                    self.__dict__.pop(bk)
            with logwatch('loading documents into memory') as lw:
                self._id_docs = list(self.iter_docs(lim=lim,lemmatize=lemmatize))
                lw.log(f'loaded {len(self._id_docs):,} documents into memory')
        return self.docs
    
    @cached_property
    def id_docs(self):
        if self._id_docs is None: self.init_docs()
        return self._id_docs
    @cached_property
    def docs(self): return [y for x,y in self.id_docs]
    @cached_property
    def ids(self): return [x for x,y in self.id_docs]
    @cached_property
    def doc2id(self): return {y:x for x,y in self.id_docs}
    @cached_property
    def id2doc(self): return {x:y for x,y in self.id_docs}


class TomotopyTopicModel(BaseTopicModel):
    model_type='tomotopy'
    def model(self, output_dir=None, force=False, lim=None, lemmatize=False, ntopic=25, niter=1000):
        with logwatch('loading or modeling LDA model') as lw:
            # get filename
            pathd = self.get_paths(output_dir=output_dir, ntopic=ntopic, niter=niter, lemmatize=lemmatize)
            fdir=pathd['path']
            os.makedirs(fdir, exist_ok=True)
            fn=pathd['path_model']
            fnindex=pathd['path_index']
            fnparams=pathd['path_params']

            # reset caches
            for attr in self.attrs_set_after_modeling:
                if attr in self.__dict__:
                    del self.__dict__[attr]

            # save?
            if force or not os.path.exists(fn) or not os.path.exists(fnindex):
                mdl = self.mdl = tp.LDAModel(k=ntopic)
                docd=self.id2index={}
                for page_id,page_txt in self.iter_docs(lim=lim,lemmatize=lemmatize,as_str=False):
                    docd[page_id] = mdl.add_doc(page_txt)

                def getdesc():
                    return f'{lw.inner_pref}training model (ndocs={len(docd)}, log-likelihood = {mdl.ll_per_word:.4})'
                
                pbar=lw.iter_progress(list(range(0, niter, 1)),desc=getdesc(),position=0)
                
                for i in pbar:
                    pbar.set_description(getdesc())
                    mdl.train(1)
                mdl.save(fn)
                lw.log(f'saved: {fn}')
                with open(fnindex,'wb') as of:
                    of.write(orjson.dumps(docd,option=orjson.OPT_INDENT_2))
                lw.log(f'saved: {fnindex}')

                params=dict(
                    ntopic=ntopic,
                    niter=niter,
                    lim=lim,
                    lemmatize=lemmatize,
                    **self.query_kwargs
                )
                with open(fnparams,'wb') as of:
                    of.write(orjson.dumps(params,option=orjson.OPT_INDENT_2))
        
                self.index2id={v:k for k,v in self.id2index.items()}

                # write docs
                write_excel(self.doc_df.reset_index(),self.path_docdf, col_widths={'page_text':80})
                write_excel(self.topic_df.reset_index(), self.path_topicdf)

            else:
                lw.log(f'loading: {fn}')
                self.mdl = tp.LDAModel.load(fn)
                with open(fnindex,'rb') as f: self.id2index=orjson.loads(f.read())
                self.index2id={v:k for k,v in self.id2index.items()}
                self.__dict__['doc_df'] = pd.read_excel(self.path_docdf).set_index(['work_id','page_id'])
            # self.mdl.summary(topic_word_top_n=25)
        
    
    @cached_property
    def topic_term_dists(self): 
        return np.stack([self.mdl.get_topic_word_dist(k) for k in range(self.mdl.k)])

    @cached_property
    def doc_topic_dists(self):
        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in self.mdl.docs])
        doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
        return doc_topic_dists


    @cached_property
    def doc_lengths(self):
        return np.array([len(doc.words) for doc in self.mdl.docs])

    @cached_property
    def vocab(self): return list(self.mdl.used_vocabs)
    @cached_property
    def term_frequency(self): return self.mdl.used_vocab_freq

    def save_pyldavis(self, output_dir=None, force=False):
        output_dir = self.path_ldavis if not output_dir else output_dir
        fn=os.path.join(output_dir, 'index.html')
        if not force and os.path.exists(fn): return fn

        with logwatch('saving pyldavis output') as lw:
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
            os.makedirs(output_dir, exist_ok=True)
            lw.log(f'saving: {fn}')
            pyLDAvis.save_html(prepared_data, fn)
        return fn
    
    def visualize_topics(self):
        from IPython.display import HTML
        ldavis_fn = self.save_pyldavis()
        return HTML(filename=ldavis_fn)
    

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
    def meta(self): return self.work_df
    
    @cached_property
    def topic_names(self, top_n=25):
        d={}
        for topic_id in range(self.mdl.k):
            d[topic_id] = f'Topic {topic_id}: {" ".join(w for w,c in self.mdl.get_topic_words(topic_id,top_n=top_n))}'
        return d

    @cached_property
    def num_topics(self): return self.mdl.k

    @cached_property
    def doc_df(self):
        with logwatch('collecting document data') as lw:
            ld=[]
            for i,doc in enumerate(lw.iter_progress(self.mdl.docs,desc='iterating over documents in model',position=0)):
                page_id = self.index2id[i]
                page_topics = doc.get_topics(top_n=self.num_topics)
                page_topics.sort(key = lambda lt: -lt[-1])
                top_topic = page_topics[0][0]
                topic_scores = dict(sorted(page_topics))
                odx={
                    'work_id':get_work_id(page_id),
                    'page_id':page_id,
                    'page_text':self.topicize_text(self.corpus[page_id].txt, top_topic),
                    'Top Topic':self.topic_names.get(top_topic),
                    'Top Topic Score':topic_scores[top_topic],
                    **{
                        self.topic_names[tnum]:tscore
                        for tnum,tscore in topic_scores.items()
                    }
                }
                ld.append(odx)
        with logwatch('merging with metadata'):    
            mdf=self.corpus.meta.reset_index()
            odf=pd.DataFrame(ld).fillna(0)
            odf=mdf.merge(odf,on='work_id')
            return odf.set_index('work_id')

    @cached_property
    def work_df(self):
        mdf=self.corpus.meta.reset_index()
        docdf=self.doc_df.groupby('work_id').mean().reset_index()
        return mdf.merge(docdf,on='work_id').set_index(list(mdf.columns))

        
    def topicize_text(self, txt, topic_id):
        words = {w for w,c in self.mdl.get_topic_words(topic_id,top_n=100)}
        return untokenize_agnostic(
            f'**{w}**' if w.lower() in words else w
            for w in tokenize_agnostic(txt)
        )
    
    @cached_property
    def topic_df(self, top_n=100):
        df=pd.DataFrame(
            dict(
                topic_id=i, 
                topic_name=self.topic_names[i],
                topic_words=' '.join(w for w,c in self.mdl.get_topic_words(i, top_n=top_n))
            )
            for i in range(self.num_topics)
        )
        return df.set_index('topic_id')






class BertTopicModel(BaseTopicModel):
    model_type='bertopic'
    embedding_model_name = 'emanjavacas/MacBERTh'

    @cached_property
    def embedding_model(self):
        from transformers.pipelines import pipeline
        embedding_model = pipeline(
            "feature-extraction", 
            model=self.embedding_model_name,
            # model_kwargs=dict(
                # max_length=512
            # )
        )
        return embedding_model
    



    def model(self, output_dir=None,force=False, lim=None, save=True, embedding_model=None, lemmatize=True, **kwargs):
        with logwatch('loading or generating model'):
            # get filename
            pathd=self.get_paths(output_dir=output_dir)
            fdir=pathd['path']
            os.makedirs(fdir, exist_ok=True)
            fn=pathd['path_model']
            if not force and os.path.exists(fn): return self.load(fn)

            with logwatch('importing BERTopic'):
                os.environ['TOKENIZERS_PARALLELISM']='false'
                from bertopic import BERTopic
                from bertopic.representation import KeyBERTInspired

            # get docs
            docs = self.init_docs(lim=lim, lemmatize=lemmatize)

            with logwatch('fitting model'):
                self._mdl = BERTopic(
                    embedding_model=embedding_model,
                    representation_model=KeyBERTInspired(), 
                    verbose=True, 
                    **kwargs
                )
                self._topics, self._probs = self._mdl.fit_transform(docs)
            self._mdl.generate_topic_labels(nr_words=10)
            if save: self.save(fn)
            return self._mdl


    def save(self, fn=None):
        if self._mdl is not None:
            fn=self.path_model if fn is None else fn
            with logwatch(f'saving model to disk: {truncfn(fn)}'):
                ensure_dir(fn)
                self.mdl.save(fn)
                write_excel(self.doc_df.reset_index(), self.path_docdf, col_widths={'page_text':80, 'document':80})
                write_excel(self.topic_df.reset_index(), self.path_topicdf, col_widths={'representative_doc1':80, 'representative_doc2':80, 'representative_doc3':80})

    def load(self, fn=None):
        if not fn: fn=self.path_model
        if os.path.exists(fn):
            with logwatch('importing BERTopic'):
                from bertopic import BERTopic
            with logwatch(f'loading model from disk: {truncfn(fn)}'):
                self._mdl = BERTopic.load(self.path_model)
                return self._mdl

    @cached_property
    def doc_df(self):
        docinfo = self.mdl.get_document_info(self.docs)
        docinfo.columns = [x.lower() for x in docinfo]
        docinfo['page_id']=[self.doc2id[doc] for doc in docinfo.document]
        docinfo = docinfo.drop('representative_docs',1)
        docinfo['page_text']=[self.corpus[id].txt for id in docinfo.page_id]
        for x in ['representation']:
            docinfo[x]=docinfo[x].apply(lambda x: ' '.join(x))
        return docinfo[[c for c in docinfo if c!='document']+['document']]


    @cached_property
    def topic_df(self):
        tdf = self.mdl.get_topic_info()
        tdf.columns = [x.lower() for x in tdf]
        tdf['representative_docs_ids']=[[self.doc2id[doc] for doc in docs] for docs in tdf.representative_docs]
        tdf = tdf.drop('representative_docs',1)
        for i in range(3):
            tdf[f'representative_doc{i+1}']=[self.corpus[reprids[i]].txt for reprids in tdf.representative_docs_ids]
        
        for x in ['representation','representative_docs_ids']:
            tdf[x]=tdf[x].apply(lambda x: ' '.join(x))
        return tdf

    @cached_property
    def page2topic(self):
        return dict(zip(self.doc_df.page_id,self.doc_df.name))
    
    @cached_property
    def hierarchical_topics(self):
        return self.mdl.hierarchical_topics(self.docs)
    
    def visualize_hierarchy(self):
        fig = self.mdl.visualize_hierarchy(hierarchical_topics=self.hierarchical_topics)
        fig.write_html(self.path_hclust)
        return fig



def PPATopicModel(model_type='bertopic',**kwargs):
    if model_type:
        mtyp = model_type.lower()
        if mtyp.startswith('tomo') or mtyp.startswith('lda'):
            return TomotopyTopicModel(**kwargs)
    return BertTopicModel(**kwargs)