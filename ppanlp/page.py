from .imports import *


class PPAPage:
    def __init__(self, id, text=None,**_meta):
        self.id = id
        self.text = text if text is not None else PPA().textd[id.split('_')[0]]
        self.corpus = text.corpus
        self._meta = _meta if _meta else text.pages_d.get(self.id)._meta

    
    
    @cached_property
    def meta(self):
        return pd.Series({
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
        }, name=self.id)
    
    @cached_property
    def db_input(page):
        return dict(
            page_id=page.id,
            page_text=page.txt,
            page_num=page.num,
            page_num_orig=page.num_orig,
            page_num_content_words=page.num_content_words,
            page_ocr_accuracy=page.ocr_accuracy,
            work_id=page.text.id,
            cluster = page.text.cluster,
            source = page.text.source,
            year = page.text.year,
            author = page.text.author,
            title = page.text.title[:255],
            _random = random.random()
        )
    
    @cached_property
    def txt(self):
        return self._meta.get('page_text','')
    
    @cached_property
    def num(self):
        return self._meta.get('page_num', int(self.id.split('.')[-1]))
    @cached_property
    def num_orig(self):
        return self._meta.get('page_num_orig','')
    
    @cached_property
    def tokens(self):
        tokens=self._meta.get('page_tokens')
        if not tokens: tokens=tokenize_agnostic(self.txt)
        tokens = [x.strip().lower() for x in tokens if x.strip() and x.strip()[0].isalpha()]
        return tokens

    @cached_property
    def words(self): 
        return [tok for tok in self.tokens if tok and any(y.isalpha() for y in tok)]
    @cached_property
    def num_words(self): return len(self.words)
    @cached_property
    def num_tokens(self): return len(self.tokens)
    @cached_property
    def num_words_recog(self): return len(self.words_recog)
    
    @cached_property
    def content_words(self): 
        return self.get_content_words()
    @cached_property
    def content_lemmas(self): 
        return [self.corpus.lemmatize(word) for word in self.content_words]
    @cached_property
    def num_content_words(self): 
        res=self._meta.get('page_num_content_words')
        if res is None: res=len(self.content_words)
        return res
    @cached_property
    def stopwords(self): return self.corpus.stopwords

    @cached_property
    def words_recog(self):
        return [w for w in self.words if w in get_english_wordlist()]

    @cached_property
    def ocr_accuracy(self):
        return self.num_words_recog / self.num_words if self.num_words else 0

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

    def get_tokens(self, orig=True, lemmatize=False, as_str=False):
        if lemmatize:
            l = self.content_lemmas
        elif orig:
            l = self.tokens
        else:
            l = self.content_words
        return ' '.join(l) if as_str else l