import os
import json
from tqdm import tqdm
from .legal-corpora.main import for_computation

PSEUDO_LABELLING_THRESHOLD = 0.3
INCLUDE_META_DATA = False
JSON_EXTENSION = ".json"
TRAINING_JSON_PATH = "./processed_document_jsons/train/"
VALIDATION_JSON_PATH = "./processed_document_jsons/valid/"
TEST_JSON_PATH = "./processed_document_jsons/test/"
RAW_DATA_PATH_HEADNOTE = "raw_files/Headnote"
RAW_DATA_PATH_INFO = "raw_files/Info"
RAW_DATA_PATH_JUDGEMENT = "raw_files/Judgement"
HEADNOTE_EXTENTION = ".headnote"
JUDGEMENT_EXTENTION = ".judgement"
INFO_FILE_EXTENTION = ".info"

Files = os.listdir("data/")
with open("./mapping.json","r") as f:
    mapping = json.load(f)

for fi in mapping['train']:
    try:
        doc = {}
        with open(RAW_DATA_PATH_HEADNOTE+fi+HEADNOTE_EXTENTION , "r") as f:
            doc['headnote'] = f.read()
        with open(RAW_DATA_PATH_INFO+fi+INFO_FILE_EXTENTION, "r") as f:
            doc['info'] = f.read()
        with open(RAW_DATA_PATH_JUDGEMENT+fi+JUDGEMENT_EXTENTION, "r") as f:
            doc['judgement'] = f.read()
    except:
        print(fi)

    for_computation(doc, INCLUDE_META_DATA, PSEUDO_LABELLING_THRESHOLD, TRAINING_JSON_PATH+fi+JSON_EXTENSION)

for fi in mapping['valid']:
    try:
        doc = {}
        with open(RAW_DATA_PATH_HEADNOTE+fi+HEADNOTE_EXTENTION , "r") as f:
            doc['headnote'] = f.read()
        with open(RAW_DATA_PATH_INFO+fi+INFO_FILE_EXTENTION, "r") as f:
            doc['info'] = f.read()
        with open(RAW_DATA_PATH_JUDGEMENT+fi+JUDGEMENT_EXTENTION, "r") as f:
            doc['judgement'] = f.read()
    except:
        print(fi)

    for_computation(doc, INCLUDE_META_DATA, PSEUDO_LABELLING_THRESHOLD, VALIDATION_JSON_PATH+fi+JSON_EXTENSION)

for fi in mapping['test']:
    try:
        doc = {}
        with open(RAW_DATA_PATH_HEADNOTE+fi+HEADNOTE_EXTENTION , "r") as f:
            doc['headnote'] = f.read()
        with open(RAW_DATA_PATH_INFO+fi+INFO_FILE_EXTENTION, "r") as f:
            doc['info'] = f.read()
        with open(RAW_DATA_PATH_JUDGEMENT+fi+JUDGEMENT_EXTENTION, "r") as f:
            doc['judgement'] = f.read()
    except:
        print(fi)

    for_computation(doc, INCLUDE_META_DATA, PSEUDO_LABELLING_THRESHOLD, TEST_JSON_PATH+fi+JSON_EXTENSION)