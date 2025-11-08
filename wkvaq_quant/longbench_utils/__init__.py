from .scorer import scorer

import json
import os

model2maxlen   = json.load(open(os.path.join(os.path.dirname(__file__), "config/model2maxlen.json"), "r"))
dataset2prompt = json.load(open(os.path.join(os.path.dirname(__file__), "config/dataset2prompt.json"),"r"))
dataset2maxlen = json.load(open(os.path.join(os.path.dirname(__file__), "config/dataset2maxlen.json"), "r"))