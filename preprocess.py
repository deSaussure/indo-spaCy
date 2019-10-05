import os
import sys
import pandas as pd
import re

DIR_this = os.path.dirname(os.path.abspath(__file__))
DIR_dataset = os.path.abspath(os.path.join(DIR_this, 'dataset', 'ner'))

sys.path.append(DIR_this)
sys.path.append(DIR_dataset)

class Preprocess():
    def __init__(self, **kwargs):
        self.ner_dataset = open(os.path.join(DIR_dataset, 'data_train.txt')).read().splitlines()
        self.pattern_get_entities = re.compile(r"(?:<[A-Z]+).+?(?=\</)")
        self.pattern_label_ent = re.compile(r"<([A-Z]+)>(.+)")
        self.clean_text = re.compile(r"<\/?[A-Z]+>")
    
    def preprocess_ner(self):
        dataset = []
        for text in self.ner_dataset:
            text = text.strip()
            get_entities = re.findall(self.pattern_get_entities, text)
            ent_label = []
            for _entlabel in get_entities:
                entlabel = re.findall(self.pattern_label_ent, _entlabel)
                entlabel = [(j[0], j[1]) for j in entlabel]
                ent_label.extend(entlabel)
            clean_text = re.sub(self.clean_text, '', text).strip()
            for ent in ent_label:
                ents = {}
                label = ent[0]
                start = clean_text.index(ent[1])
                end = start + len(ent[1]) + 1
                ents["entities"] = [(start, end, label)]
                dataset.append((clean_text, ents))
        return dataset

if __name__ == "__main__":
    helper = Preprocess()
    helper.preprocess_ner()
            
            