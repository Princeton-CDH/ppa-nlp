from .imports import *
import multiprocessing as mp

def iter_json(fn):
    if 'jsonl' in fn:
        yield from iter_jsonl(fn)
    else:
        yield from read_json(fn)

def iter_jsonl(fn):
    if os.path.exists(fn):
        yield from orjsonl.stream(fn)
                
def write_jsonl(obj, fn):
    ensure_dir(fn)
    orjsonl.save(fn, obj)

def read_jsonl(fn):
    return list(iter_jsonl(fn))


def read_json(fn):
    if os.path.exists(fn):
        try:
            if 'jsonl' in fn:
                return read_jsonl(fn)
            if fn.endswith('.gz'):
                try:
                    with gzip.open(fn, 'r') as zipfile:
                        return orjson.loads(zipfile.read())
                except gzip.BadGzipFile:
                    pass
            with open(fn, 'rb') as f:
                return orjson.loads(f.read())
        except Exception as e:
            # print('!!',fn,e)
            pass    
    return []

def write_json(obj, fn):    
    if 'jsonl' in fn: return write_jsonl(obj, fn)
    ensure_dir(fn)
    
    if fn.endswith('.gz'):
        with gzip.open(fn, 'w') as zipfile:
            zipfile.write(orjson.dumps(obj))
    else:
        with open(fn, 'wb') as of:
            of.write(orjson.dumps(obj,option=orjson.OPT_INDENT_2))


def tokenize_agnostic(txt):
    return re.findall(r"[\w']+|[.,!?; -—–'\n]", txt)

def untokenize_agnostic(l):
    return ''.join(l)

def ensure_dir(fn):
    dirname=os.path.dirname(fn)
    if dirname: os.makedirs(dirname, exist_ok=True)
    
def get_num_lines_json(fn, progress=True):
    if not os.path.exists(fn): return
    nl=sum(1 for _ in tqdm(iter_json(fn), desc=f'Counting lines in {fn[fn.index(".json"):] if ".json" in fn else fn}',disable=not progress,position=0))
    return nl




# Helper Progress Iterator
# Needs: python -m pip install enlighten
# https://stackoverflow.com/a/63796670
def piter(it, *pargs, **nargs):
    import enlighten
    global __pit_man__
    try:
        __pit_man__
    except NameError:
        __pit_man__ = enlighten.get_manager()
    man = __pit_man__
    try:
        it_len = len(it)
    except:
        it_len = None
    try:
        ctr = None
        for i, e in enumerate(it):
            if i == 0:
                ctr = man.counter(*pargs, **{**dict(leave = False, total = it_len), **nargs})
            yield e
            ctr.update()
    finally:
        if ctr is not None:
            ctr.close()

def encode_cache(x): return zlib.compress(orjson.dumps(x,option=orjson.OPT_SERIALIZE_NUMPY))
def decode_cache(x): return orjson.loads(zlib.decompress(x))

def CompressedSqliteDict(fn, *args, flag='c', **kwargs):
    kwargs['encode']=encode_cache
    kwargs['decode']=decode_cache
    if not os.path.exists(fn) and flag=='r':
        ensure_dir(fn)
        with CompressedSqliteDict(fn, flag='c') as db: pass
        
    return SqliteDict(fn, *args, flag=flag, **kwargs)





CONTEXT='fork'
# default num proc is?
mp_cpu_count=mp.cpu_count()
if mp_cpu_count==1: DEFAULT_NUM_PROC=1
elif mp_cpu_count==2: DEFAULT_NUM_PROC=2
elif mp_cpu_count==3: DEFAULT_NUM_PROC=2
else: DEFAULT_NUM_PROC = mp_cpu_count - 2


def in_jupyter(): return sys.argv[-1].endswith('json')

def get_tqdm(iterable,*args,**kwargs):
    return tqdm(iterable, **kwargs)



def pmap_iter(
        func, 
        objs, 
        args=[], 
        kwargs={}, 
        lim=None,
        num_proc=DEFAULT_NUM_PROC, 
        use_threads=False, 
        progress=True, 
        progress_pos=0,
        desc=None,
        shuffle=False,
        context=CONTEXT, 
        **y):
    """
    Yields results of func(obj) for each obj in objs
    Uses multiprocessing.Pool(num_proc) for parallelism.
    If use_threads, use ThreadPool instead of Pool.
    Results in any order.
    """

    # lim?
    if shuffle: random.shuffle(objs)
    if lim: objs = objs[:lim]

    # check num proc
    num_cpu = mp.cpu_count()
    if num_proc>num_cpu: num_proc=num_cpu
    if num_proc>len(objs): num_proc=len(objs)

    # if parallel
    if not desc: desc=f'Mapping {func.__name__}()'
    if desc and num_cpu>1: desc=f'{desc} [x{num_proc}]'
    if num_proc>1 and len(objs)>1:

        # real objects
        objects = [(func,obj,args,kwargs) for obj in objs]

        # create pool
        #pool=mp.Pool(num_proc) if not use_threads else mp.pool.ThreadPool(num_proc)
        with mp.get_context(context).Pool(num_proc) as pool:
            # yield iter
            iterr = pool.imap(_pmap_do, objects)

            for res in get_tqdm(iterr,total=len(objects),desc=desc,position=progress_pos) if progress else iterr:
                yield res
    else:
        # yield
        for obj in (tqdm(objs,desc=desc,position=progress_pos) if progress else objs):
            yield func(obj,*args,**kwargs)

def _pmap_do(inp):
    func,obj,args,kwargs = inp
    return func(obj,*args,**kwargs)

def pmap(*x,**y):
    """
    Non iterator version of pmap_iter
    """
    # return as list
    return list(pmap_iter(*x,**y))

def pmap_run(*x,**y):
    for obj in pmap_iter(*x,**y): pass




class logwatch:
    """A class for monitoring and logging the duration of tasks.

    Attributes:
        started (float): The timestamp when the task started.
        ended (float): The timestamp when the task ended.
        level (str): The logging level for the task. Default is 'DEBUG'.
        log (Logger): The logger object for logging the task status.
        task_name (str): The name of the task being monitored.
    """
    def __init__(self, name='running task', level='DEBUG'):
        self.started = None
        self.ended = None
        self.level=level
        self.log = getattr(logger,self.level.lower())
        self.task_name = name
    @property
    def tdesc(self): 
        """Returns the formatted timespan of the duration.
        
        Returns:
            str: The formatted timespan of the duration.
        
        Examples:
            >>> t = tdesc(self)
            >>> print(t)
            '2 hours 30 minutes'
        """
        return format_timespan(self.duration)
    
    @property
    def duration(self): 
        """Calculates the duration of an event.
        
        Returns:
            float: The duration of the event in seconds.
        """
        return self.ended - self.started
    
    @property
    def desc(self): 
        """Returns a description of the task.
        
        If the task has both a start time and an end time, it returns a string
        indicating the task name and the time it took to complete the task.
        
        If the task is currently running, it returns a string indicating that
        the task is still running.
        
        Returns:
            str: A description of the task.
        """
        if self.started is not None and self.ended is not None:
            return f'finished {self.task_name} in {self.tdesc}'
        else:
            return f'Task running ...' if not self.task_name else f'{self.task_name} ...'
        
    def __enter__(self):        
        """Context manager method that is called when entering a 'with' statement.
        
        This method logs the description of the context manager and starts the timer.
        
        Examples:
            with Logwatch():
                # code to be executed within the context manager
        """
        self.log(self.desc)
        self.started = time.time()
        return self

    def __exit__(self,*x):
        """
        Logs the resulting time.
        """ 
        self.ended = time.time()
        self.log(self.desc)

