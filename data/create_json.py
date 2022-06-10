import os
import json
from tqdm import tqdm

Files = os.listdir("processed_document_jsons/")
with open("./mapping.json","r") as f:
    mapping = json.load(f)

train_json = {}
test_json = {}
val_json = {}

for fi in mapping['train']:
    try:
        with open("processed_document_jsons/train"+fi+".json", "r") as f:
            doc = json.load(f)
        train_json[fi] = doc
    except:
        print(fi)

with open("./nl_data/train.json","w") as f:
    json.dump(train_json,f)

for fi in mapping['valid']:
    try:
        with open("processed_document_jsons/valid"+fi+".json", "r") as f:
            doc = json.load(f)
        val_json[fi] = doc
    except:
        print(fi)

with open("./nl_data/valid.json","w") as f:
    json.dump(val_json,f)

for fi in mapping['test']:
    try:
        with open("processed_document_jsons/test"+fi+".json", "r") as f:
            doc = json.load(f)
        test_json[fi] = doc
    except:
        print(fi)

with open("./nl_data/test.json","w") as f:
    json.dump(test_json,f)

