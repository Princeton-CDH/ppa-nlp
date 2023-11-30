# imports...
import os,sys,warnings,random
warnings.filterwarnings('ignore')
from functools import cache
from tqdm import tqdm
from sqlitedict import SqliteDict
import orjson,zlib
import pandas as pd
from intspan import intspan
import jsonlines

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
nltk.download('punkt')

## settings
pd.options.display.max_columns=None

## constants
PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_REPO = os.path.dirname(PATH_HERE)
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
PATH_OCR_RULESETS = os.path.join(PATH_REPO_DATA, 'ocr_cleanup_rulesets')

PATH_JSON_CORPUS_ROOT = os.path.join(PATH_HOME_DATA,'corpus_json1')
PATH_JSON_CORPUS_TEXTS = os.path.join(PATH_JSON_CORPUS_ROOT, 'texts')
PATH_JSON_CORPUS_METADATA = os.path.join(PATH_JSON_CORPUS_ROOT, 'metadata.csv')
PATH_TEXT_CORPUS_MINI = os.path.join(PATH_TEXT_CORPUS_ROOT, 'PPA_pages.jsonl')
## ensure dirs
for pathstr in [PATH_HOME_DATA, PATH_ECCO_DATA, PATH_ECCO_RAW_DATA]:
    os.makedirs(pathstr, exist_ok=True)