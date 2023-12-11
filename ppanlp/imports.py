# imports...
import os,sys,warnings,random
from string import punctuation
from functools import cache
from tqdm import tqdm
from sqlitedict import SqliteDict
import orjson
import pickle
import zlib
import pandas as pd
from intspan import intspan
import jsonlines
import json
import orjsonl
import time
import click
import multiprocessing as mp


## ocr correction imports
import re
import wordfreq
import os
import json
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from collections import defaultdict
from difflib import SequenceMatcher
from functools import cached_property
from collections import Counter
import gzip
import numpy as np
from humanfriendly import format_timespan as ftspan

def format_timespan(*args,replace={'0 seconds':'0.0 seconds'},**kwargs):
    res = ftspan(*args,**kwargs)
    return replace.get(res,res)



# nltk.download('punkt')

## settings
pd.options.display.max_columns=None
pd.options.display.max_rows=10

## constants
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_REPO = os.path.dirname(PATH_HERE)
PATH_REPO_CODE = os.path.join(PATH_REPO,'ppanlp')
PATH_REPO_DATA = os.path.join(PATH_REPO, 'data')
PATH_HOME_DATA = os.path.expanduser('~/ppa_data')
PATH_ECCO_DATA = os.path.join(PATH_HOME_DATA, 'ecco')
PATH_ECCO_RAW_DATA = os.path.join(PATH_ECCO_DATA, 'raw')
PATH_CORPUS_RAW_ECCO1=os.path.join(PATH_ECCO_RAW_DATA, 'ECCO1')
PATH_CORPUS_RAW_ECCO2=os.path.join(PATH_ECCO_RAW_DATA, 'ECCO2')
PATH_ECCO_PAGES_CACHE = os.path.join(PATH_ECCO_DATA,'ecco_pages.sqlitedict')
PATH_PPA_PAGES_CACHE = os.path.join(PATH_HOME_DATA,'ppa_pages.sqlitedict')
PATH_PPA_PAGES_CACHE_KEYS = os.path.join(PATH_HOME_DATA,'ppa_pages.sqlitedict.keys')
# PATH_ECCO_EXCERPTS = os.path.join(PATH_REPO_DATA,'ECCO Excerpts-DigitizedWork-2023-09-20.csv')
PATH_PPA_METADATA = os.path.join(PATH_REPO_DATA,'ALL-DigitizedWork-2023-09-14.csv')
PATHS_METADATA = [
    os.path.join(PATH_REPO_DATA, 'data.all_xml_metadata.ECCO1.csv'), 
    os.path.join(PATH_REPO_DATA, 'data.all_xml_metadata.ECCO2.csv')
]
PATH_HATHI_PAGE_JSONL = os.path.join(PATH_REPO_DATA, 'data.hathi_pages.jsonl')
PATH_TEXT_CORPUS_ROOT = os.path.join(PATH_HOME_DATA, 'corpus')
PATH_TEXT_CORPUS_TEXTS = os.path.join(PATH_TEXT_CORPUS_ROOT, 'texts')
PATH_TEXT_CORPUS_METADATA = os.path.join(PATH_TEXT_CORPUS_ROOT, 'metadata.csv')
PATH_OCR_RULESETS = os.path.join(PATH_REPO_CODE, 'ocr_cleanup_rulesets')

PATH_JSON_CORPUS_ROOT = os.path.join(PATH_HOME_DATA,'corpus_json1')
PATH_JSON_CORPUS_TEXTS = os.path.join(PATH_JSON_CORPUS_ROOT, 'texts')
PATH_JSON_CORPUS_METADATA = os.path.join(PATH_JSON_CORPUS_ROOT, 'metadata.csv')
PATH_TEXT_CORPUS_MINI = os.path.join(PATH_TEXT_CORPUS_ROOT, 'PPA_pages.jsonl')
PATH_PPA_CORPUS = os.path.join(PATH_HOME_DATA,'corpus')

GROUP_KEY='group_id_s'
CLUSTER_KEY='cluster_id_s'
SOURCE_KEY='source_id'
WORK_KEY='work_id'
PAGE_KEY='page_id'

## ensure dirs
for pathstr in [PATH_HOME_DATA, PATH_ECCO_DATA, PATH_ECCO_RAW_DATA]:
    os.makedirs(pathstr, exist_ok=True)

# setup logs
# LOG_FORMAT = '<green>{time:YYYY-MM-DD HH:mm:ss,SSS}</green> - <cyan>{function}</cyan> - <level>{message}</level> - <cyan>{file}</cyan>:<cyan>{line}</cyan>'
# LOG_FORMAT = '<green>{time:YYYY-MM-DD HH:mm:ss,SSS}</green> <level>{message}</level>'
LOG_FORMAT = '<level>{message}</level> @ <green>{time:YYYY-MM-DD HH:mm:ss,SSS}</green>'

# 5 to include traces; 
# 10 for debug; 20 info, 25 success; 
# 30 warning, 40 error, 50 critical;
LOG_LEVEL = 10

from loguru import logger
logger.remove()
logger.add(
    sink=sys.stderr,
    format=LOG_FORMAT, 
    level=LOG_LEVEL
)
from .utils import *
from .cleanup import *
from .corpus import *
from .text import *
from .page import *
from .subcorpus import *
from .topicmodel import *
from .ner import *
from .cli import *

warnings.filterwarnings('ignore')