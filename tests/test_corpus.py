import os,sys; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ppanlp.corpus import *

def test_PPA():
    a=PPA()
    b=PPA()
    set_corpus_path('./otherdir')
    c=PPA()
    d=PPA('./anotherdir')

    assert a is b
    assert a is not c
    assert a is not d
    assert c is not d