from .imports import *
from scipy.stats.contingency import odds_ratio
from scipy.stats import fisher_exact


class NERModel:
    def __init__(self, corpus=None):
        self.corpus = corpus if corpus is not None else PPA()
        self.ent2pages=defaultdict(set)
        self.ent2count=Counter()
        self._annodf=None
        
    @staticmethod
    def clean_ent(ent):
        o=ent.strip(punctuation).title()
        if o.endswith("'S"): o = o[:-2]
        return o
    
    @staticmethod
    def iter_by_page(iterr):
        last_pageid=None
        last_l=[]
        for resd in iterr:
            pageid=resd['page_id']
            if last_l and pageid!=last_pageid:
                yield last_l
                last_l=[]
            last_l.append(resd)
            last_pageid = pageid
        if last_l: yield last_l
        
    def iter_ents(self,ent_types:set=None,lim=None,by_page=False,ents=None):
        def iterr():
            with self.corpus.ents_db(flag='r') as db:
                total=len(db)
                iterr=tqdm(db.items(),desc='Iterating over saved ents',position=0,total=total)
                for page_id,page_ents in iterr:
                    for ent,ent_type in page_ents:
                        ent = self.clean_ent(ent)
                        if (not ent_types or ent_type in ent_types) and (not ents or ent in ents):
                            yield {'page_id':page_id,'ent':self.anno_ents.get(ent,ent),'ent_type':ent_type,'ent_orig':ent}
        oiterr = (self.iter_by_page(iterr()) if by_page else iterr())
        yield from iterlim(oiterr,lim)
    
                            
    def iter_persons(self, **kwargs):
        kwargs['ent_types']={'PERSON'}
        yield from self.iter_ents(**kwargs)
    
    def count_ents(self, **kwargs):
        self.ent2count=Counter()
        for resd in self.iter_ents(**kwargs):
            page_id,ent = resd['page_id'],resd['ent']
            self.ent2pages[ent].add(page_id)
            self.ent2count[ent]+=1
        self.ent2count_s = pd.Series(self.ent2count).sort_values(ascending=False)
        return self.ent2count_s
    
    def count_persons(self, **kwargs):
        kwargs['ent_types']={'PERSON'}
        return self.count_ents(**kwargs)
    
    def prep_anno_df(self, min_count=100):
        s = ner.ent2count_s
        s = s[s>=min_count]
        df = pd.DataFrame({'count':s}).rename_axis('name')
        df['is_valid'] = ''
        return df
    
    @cached_property
    def path_to_anno(self): return os.path.join(self.corpus.path_data, 'data.ner.to_anno.csv')
    @cached_property
    def path_anno(self): return os.path.join(self.corpus.path_data, 'data.ner.anno.csv')
    
    def load_anno_df(self, fn=None, force=False):
        if force or self._annodf is None:
            fn=fn if fn else self.path_anno
            self._annodf = pd.read_csv(fn).set_index('name').fillna('')
        return self._annodf
    
    @cached_property
    def anno_df(self): return self.load_anno_df()

    @cached_property
    def anno_ents(self): 
        df=self.anno_df
        df=df[df.is_valid.str.startswith('y')]
        return {
            k:(v if v else k)
            for k,v in zip(df.index, df.who)
        }

    def iter_ents_anno(self, **kwargs):
        kwargs['ents']=set(self.anno_ents.keys())
        yield from self.iter_ents(**kwargs)

    def iter_persons_anno(self, **kwargs):
        kwargs['ent_types']={'PERSON'}
        yield from self.iter_ents_anno(**kwargs)
        
    @cached_property
    def persons_anno_pagedata(self):
        return [pdata for pdata in self.iter_persons_anno(by_page=True) if pdata]
            
    def person_cooccurence(self, min_page_count=25, lim=None, topic_model = None, funcs = [odds_ratio, fisher_exact], **kwargs):
        data = self.persons_anno_pagedata
        data = (
            random.sample(data,lim) 
            if lim and len(data)>lim 
            else data
        )
        toppref='TOPIC_'
        aupref='AUTHOR_'
        prpref='PERIOD_'
        person1 = Counter()
        person2 = Counter()
        pair_pages = defaultdict(set)
        numpages=0
        allppl=set()
        for pagedata in data:
            pageid=pagedata[0]['page_id']
            if topic_model and not pageid in topic_model.page2topic: continue
            numpages+=1
            workid='.'.join(pageid.split('.')[:-1])
            text = self.corpus.textd[workid]
            pageppl = {d['ent'] for d in pagedata}

            # if text.author: 
                # pageppl.add(aupref+text.author)

            if text.year>0:
                prd=50
                prd1=text.year//prd*prd
                prd2=prd1+prd
                pageppl.add(f'{prpref}{prd1}-{prd2}')

            if topic_model:
                topic=topic_model.page2topic.get(pageid)
                if topic and not topic.startswith('-1'):
                    pageppl.add(toppref+topic)

            for x in pageppl: 
                person1[x]+=1
                allppl.add(x)
                for y in pageppl:
                    if x<y:
                        person2[x,y]+=1

        def count_ind(x):
            return person1[x]
        def count_solo(x,y):
            return person1[x] - person2[x,y] - person2[y,x]
        def count_together(x,y):
            return person2[x,y]
        def count_neither(x,y):
            return numpages - count_together(x,y) - count_solo(x,y) - count_solo(y,x)

        person1sum=sum(person1.values())
        def prob_ind(x):
            return person1[x]/numpages
        def prob_obs(x,y):
            return person2[x,y]/numpages
        def prob_exp(x,y):
            return prob_ind(x) * prob_ind(y)
        def prob_obsexp(x,y):
            return prob_obs(x,y) / prob_exp(x,y)

        def gettype(x):
            if x.startswith(toppref): return 'Topic'
            if x.startswith(aupref): return 'Author'
            if x.startswith(prpref): return 'Period'
            return 'Person'

        def get_contingency_table(x,y):
            tl=count_together(x,y)
            tr=count_solo(x,y)
            bl=count_solo(y,x)
            br=count_neither(x,y)
            return ((tl,tr),(bl,br))

        def iter_res():
            minc=min_page_count
            cmps = [
                (x,y) 
                for x in allppl 
                for y in allppl 
                if x<y 
                and count_ind(x)>=minc 
                and count_ind(y)>=minc
                and not (x.startswith(aupref) and y.startswith(prpref))
                and not x.lower() in y.lower()
                and not y.lower() in x.lower()
            ]
            for x,y in tqdm(cmps,desc='Iterating over statistical comparisons',position=0):
                val_d={
                    'entity_x':x, 
                    'entity_y':y, 
                    'type_x':gettype(x),
                    'type_y':gettype(y),
                    'num_total_x':count_ind(x), 
                    'num_total_y':count_ind(y), 
                    'num_solo_x':count_solo(x,y), 
                    'num_solo_y':count_solo(y,x),
                    'num_both_xy':count_together(x,y), 
                    'num_neither_xy':count_neither(x,y), 
                    'prob_x':prob_ind(x)*100,
                    'prob_y':prob_ind(y)*100,
                    'prob_xy_obs':prob_obs(x,y)*100,
                    'prob_xy_exp':prob_exp(x,y)*100,
                    'prob_xy_obsexp':prob_obsexp(x,y)*100,
                }
                ctbl=get_contingency_table(x,y)
                for func in funcs:
                    res = func(ctbl)
                    method=func.__name__
                    stat=res.statistic if hasattr(res,'statistic') else None
                    pval=res.pvalue if hasattr(res,'pvalue') else None
                    if stat is not None: val_d[f'{method}'] = stat
                    if pval is not None: val_d[f'{method}_p'] = pval
                if val_d.get('fisher_exact_p',1)!=1: 
                    yield val_d
        o=list(iter_res())
        if not o: return pd.DataFrame()
        odf = pd.DataFrame(o).query('fisher_exact_p!=1').sort_values('odds_ratio',ascending=False)
        odf['prob_xy_obsexp_log']=odf['prob_xy_obsexp'].apply(np.log10)
        odf=odf.replace([np.inf, -np.inf], np.nan)
        odf=odf[~odf.odds_ratio.isna()]
        return odf





