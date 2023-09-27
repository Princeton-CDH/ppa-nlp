# imports...
import os,sys,warnings,random
warnings.filterwarnings('ignore')
from functools import cache
from tqdm import tqdm
from sqlitedict import SqliteDict
import orjson,zlib
import pandas as pd
from intspan import intspan
pd.options.display.max_columns=None

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_REPO = os.path.dirname(PATH_HERE)
PATH_REPO_DATA = os.path.join(PATH_REPO, 'data')
PATH_HOME_DATA = os.path.expanduser('~/ppa_data')
PATH_ECCO_DATA = os.path.join(PATH_HOME_DATA, 'ecco')
PATH_ECCO_RAW_DATA = os.path.join(PATH_ECCO_DATA, 'raw')
PATH_CORPUS_RAW_ECCO1=os.path.join(PATH_ECCO_RAW_DATA, 'ECCO1')
PATH_CORPUS_RAW_ECCO2=os.path.join(PATH_ECCO_RAW_DATA, 'ECCO2')
PATH_ECCO_PAGES_CACHE = os.path.join(PATH_ECCO_DATA,'ecco_pages.sqlitedict')
PATH_ECCO_EXCERPTS = os.path.join(PATH_REPO_DATA,'ECCO Excerpts-DigitizedWork-2023-09-20.csv')
PATHS_METADATA = [
    os.path.join(PATH_REPO_DATA, 'data.all_xml_metadata.ECCO1.csv'), 
    os.path.join(PATH_REPO_DATA, 'data.all_xml_metadata.ECCO2.csv')
]


for pathstr in [PATH_HOME_DATA, PATH_ECCO_DATA, PATH_ECCO_RAW_DATA]:
    os.makedirs(pathstr, exist_ok=True)