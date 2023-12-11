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
            else:
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

def get_num_lines(fn):
    if os.path.exists(fn):
        with logwatch(f'counting lines in {os.path.basename(fn)}'):
            with open(fn) as f:
                return sum(1 for _ in f)
    return 0

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
        with CompressedSqliteDict(fn, *args, flag='c', **kwargs) as db: pass
        
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





NUM_LOGWATCHES=0
LOGWATCH_ID=0

class logwatch:
    """A class for monitoring and logging the duration of tasks.

    Attributes:
        started (float): The timestamp when the task started.
        ended (float): The timestamp when the task ended.
        level (str): The logging level for the task. Default is 'DEBUG'.
        log (Logger): The logger object for logging the task status.
        task_name (str): The name of the task being monitored.
    """
    def __init__(self, name='running task', level='DEBUG', min_seconds_logworthy=None):
        global LOGWATCH_ID
        LOGWATCH_ID+=1
        self.id = LOGWATCH_ID
        self.started = None
        self.ended = None
        self.level=level
        self.task_name = name
        self.min_seconds_logworthy = min_seconds_logworthy
        self.vertical_char = '￨'
        self.last_lap = None

    def log(self, msg, pref=None, inner_pref=True,level=None):
        if msg:
            logfunc = getattr(logger,(self.level if not level else level).lower())
            logfunc(f'{(self.inner_pref if inner_pref else self.pref) if pref is None else pref}{msg}')

    def iter_progress(self, iterator, desc='iterating', pref=None, position=0, **kwargs):
        desc=f'{self.inner_pref if pref is None else pref}{desc if desc is not None else ""}'
        self.pbar = tqdm(iterator,desc=desc,position=position,**kwargs)
        return self.pbar
    
    def set_progress_desc(self, desc,pref=None,**kwargs):
        if desc:
            desc=f'{self.inner_pref if pref is None else pref}{desc if desc is not None else ""}'
            self.pbar.set_description(desc,**kwargs)

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
    
    def lap(self):
        self.last_lap = time.time()
    
    @property
    def lap_duration(self):
        return time.time() - self.last_lap if self.last_lap else 0
    
    @property
    def lap_tdesc(self):
        return format_timespan(self.lap_duration)
    

    @property
    def duration(self): 
        """Calculates the duration of an event.
        
        Returns:
            float: The duration of the event in seconds.
        """
        return self.ended - self.started
    
    @cached_property
    def pref(self):
        return f"{self.vertical_char} " * (self.num-1)
    @cached_property
    def inner_pref(self):
        return f"{self.vertical_char} " * (self.num)
    

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
        if self.started is None or self.ended is None:
            return f'{self.task_name}'.strip()
        else:
            return f'⎿ {self.tdesc}'.strip()
        
    def __enter__(self):        
        """Context manager method that is called when entering a 'with' statement.
        
        This method logs the description of the context manager and starts the timer.
        
        Examples:
            with Logwatch():
                # code to be executed within the context manager
        """
        global NUM_LOGWATCHES
        NUM_LOGWATCHES+=1
        self.num = NUM_LOGWATCHES
        self.log(self.desc, inner_pref=False)
        self.started = self.last_lap = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Logs the resulting time.
        """ 
        global NUM_LOGWATCHES, LOGWATCH_ID

        if exc_type:
            LOGWATCH_ID=0
            NUM_LOGWATCHES=0
            # logger.error(f'{exc_type.__name__} {exc_value}')
            self.log(f'{exc_type.__name__} {exc_value}', level='error')
        else:
            NUM_LOGWATCHES-=1
            self.ended = time.time()
            if not self.min_seconds_logworthy or self.duration>=self.min_seconds_logworthy:
                if self.tdesc!='0 seconds':
                    self.log(self.desc, inner_pref=False)
            if NUM_LOGWATCHES==0: LOGWATCH_ID=0






def clean_filename(filename, whitelist=None, replace=' '):
    """
    Url: https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
    """
    import unicodedata
    import string

    valid_filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    whitelist = whitelist if whitelist is not None else valid_filename_chars
    char_limit = 255

    # replace spaces
    for r in replace:
        filename = filename.replace(r,'_')
    
    # keep only valid ascii chars
    cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
    
    # keep only whitelisted chars
    cleaned_filename = ''.join(c for c in cleaned_filename if c in whitelist)
    if len(cleaned_filename)>char_limit:
        print("Warning, filename truncated because it was over {}. Filenames may no longer be unique".format(char_limit))
    return cleaned_filename[:char_limit]    



def truncfn(ifn, lim=155):
    fn=ifn[1:] if ifn.startswith(os.path.sep) else ifn
    dirs=fn.split(os.path.sep)
    while len(dirs)>2 and len(fn)>lim:
        if dirs[1]=='...': dirs.pop(1)
        dirs[1]='...'
        fn=os.path.sep.join(dirs)
    return (ifn[0] if ifn.startswith(os.path.sep) else '')+fn[-lim:]


def iterlim(iterr, lim=None):
    for i,x in enumerate(iterr):
        if lim and i>=lim: break
        yield x


def read_df(path):
    if '.json' in path: return pd.DataFrame(read_json(path))
    if '.csv' in path: return pd.read_csv(path)
    if '.pkl' in path: return pd.read_pickle(path)
    raise Exception(f'what type of file is this: {path}')


def printm(x):
    from IPython.display import Markdown, display
    display(Markdown(x))

def write_excel(df,fn,**kwargs):
    return df.to_excel(fn)

# def write_excel(df, fn, wrap=True, col_width=20, col_widths={}):
#     # Create a Pandas Excel writer using XlsxWriter as the engine.
#     with pd.ExcelWriter(fn, engine='xlsxwriter') as writer:

#         # Convert the dataframe to an XlsxWriter Excel object.
#         df.to_excel(writer, sheet_name='data')
    
#         workbook  = writer.book
#         cell_format = workbook.add_format({'text_wrap': True})
#         header_format = workbook.add_format({
#             'bold': True,
#             'text_wrap': True,
#             'valign': 'top',
#             'fg_color': '#f2f2f2',
#             'border': 1}
#         )
#         worksheet = writer.sheets['data']

#         for col_num,col_name in enumerate(df.columns):
#             worksheet.set_column(col_num, col_num+1, col_widths.get(col_name,col_width), cell_format)
#             worksheet.write(0, col_num, col_name, header_format)
#         for row_num,(rowid,row) in enumerate(df.iterrows()):
#             for col_num,col_name in enumerate(df.columns):
#                 cell_value = row[col_name]
#                 if not type(cell_value) in {str,float,np.float64,int}:
#                     cell_value = json.dumps(cell_value)
#                 worksheet.write(row_num+1, col_num, cell_value, cell_format)

#     # Close the Pandas Excel writer and output the Excel file.
#     writer.close()


# def write_excel(df, fn):
#     import pandas as pd
#     from pandas.io.excel._xlsxwriter import XlsxWriter
#     from openpyxl.styles import Font
#     import re

#     class RichExcelWriter(XlsxWriter):
#         def __init__(self, *args, **kwargs):
#             super(RichExcelWriter, self).__init__(*args, **kwargs)

#         def _value_with_fmt(self, val):
#             if type(val) == list:
#                 return val, None
#             return super(RichExcelWriter, self)._value_with_fmt(val)

#         def _write_cells(self, cells, sheet_name=None, startrow=0, startcol=0, freeze_panes=None):
#             sheet_name = self._get_sheet_name(sheet_name)
#             if sheet_name in self.sheets:
#                 wks = self.sheets[sheet_name]
#             else:
#                 wks = self.book.add_worksheet(sheet_name)
#                 wks.add_write_handler(list, lambda worksheet, row, col, list, style: worksheet._write_rich_string(row, col, *list))
#                 self.sheets[sheet_name] = wks
#             super(RichExcelWriter, self)._write_cells(cells, sheet_name, startrow, startcol, freeze_panes)

#     # Define the words to be made bold
#     normal_font = Font(bold=False, italic=False)
#     bolditalic_font = Font(bold=True, italic=True)
#     bold_font = Font(bold=True, italic=False)
#     italic_font = Font(bold=False, italic=True)

#     # Create a list to hold formatted rows
#     formatted_rows = []

#     # Iterate through each row in the DataFrame
#     for _, row in df.iterrows():
#         formatted_row = []
#         for cell_value in row:
#             if isinstance(cell_value, str):
#                 formatted_cell_value = []
#                 words = cell_value.split()
#                 for word in words:
#                     if word.startswith('***') and word.endswith('***'):
#                         formatted_cell_value.extend([bolditalic_font, word[3:-3]])
#                     elif word.startswith('**') and word.endswith('**'):
#                         formatted_cell_value.extend([bold_font, word[2:-2]])
#                     elif word.startswith('*') and word.endswith('*'):
#                         formatted_cell_value.extend([italic_font, word[1:-1]])
#                     else:
#                         formatted_cell_value.extend([normal_font,word])
#                 formatted_row.append(formatted_cell_value)
#             else:
#                 formatted_row.append(cell_value)
#         formatted_rows.append(formatted_row)

#     # Create a new DataFrame with the formatted rows
#     formatted_df = pd.DataFrame(formatted_rows, columns=df.columns)
#     writer = RichExcelWriter(fn)
#     formatted_df.to_excel(writer, sheet_name='Sample', index=False)
#     writer.save()


@cache
def get_english_wordlist(fn='english_wordlist.txt'):
    fnfn=os.path.join(PATH_OCR_RULESETS,fn)
    if os.path.exists(fnfn):
        with open(fnfn) as f:
            return set(f.read().split())
    else:
        return set()


def nowstr():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def hashstr(input_string, length=None):
    import hashlib
    # Create a SHA-256 hash of the input string
    sha256_hash = hashlib.sha256(input_string.encode()).hexdigest()
    # Truncate the hash to the specified length
    return sha256_hash[:length]



def get_num_waiting(resl):
    nw=0
    for id,res in resl:
        try:
            if res.successful():
                pass
        except ValueError:
            nw+=1
    return nw

def mp_iter_finished_res(resl, errors=True):
    for id,res in resl:
        try:
            if res.successful():
                yield id,res
            elif errors:
                yield id,res
        except ValueError:
            pass


def printm(x):
    from IPython.display import display,Markdown
    display(Markdown(x))