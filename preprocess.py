import os, sys
import spacy
import numpy as np
import pandas as pd

import re


DIR_this = os.path.dirname(os.path.abspath(__file__))
DIR_dataset_ner = os.path.abspath(os.path.join(DIR_this, 'dataset', 'ner'))
DIR_dataset_tagger = os.path.abspath(os.path.join(DIR_this, 'dataset', 'tagger'))

# sys.path.append(DIR_this)
sys.path.append([DIR_this, DIR_dataset_ner, DIR_dataset_tagger])

class Preprocess():
    def __init__(self, **kwargs):
        self.ner_dataset = open(os.path.join(DIR_dataset, 'data_train.txt')).read().splitlines()
        self.pattern_get_entities = re.compile(r"(?:<[A-Z]+).+?(?=\</)")
        self.pattern_label_ent = re.compile(r"<([A-Z]+)>(.+)")
        self.clean_text = re.compile(r"<\/?[A-Z]+>")

        self.tagger_dataset = pd.read_csv(os.path.join(DIR_dataset_tagger, \
                           'Indonesian_Manually_Tagged_Corpus_ID.tsv'), sep="\t", \
                            names=["word", "tag"])
        self.TAG_LABEL = {'NN':{"pos": "NOUN"}, 'SC':{"pos": "ADP"}, 'VB':{"pos": "VERB"},\
                          'NNP':{"pos": "PROPN"}, 'JJ':{"pos": "ADJ"}, 'RB':{"pos": "ADV"},\
                          'IN':{"pos": "PRP"}, 'Z':{"pos": "PUNCT"}, 'CD':{"pos": "SCONJ"},\
                          'CC':{"pos": "CONJ"}, 'PR':{"pos": "PROPN"}, 'PRP':{"pos": "PART"},\
                          'MD':{"pos": "AUX"}, 'FW':{"pos": "FOREIGN"}, 'NEG':{"pos": "NEG"},\
                          'DT':{"pos": "DET"}, 'NND':{"pos": "NN"}, 'SYM':{"pos": "SYM"},\
                          'RP':{"pos": "NUM"}, 'OD':{"pos": "OD"},'X':{"pos": "UNK"},\
                          'WH':{"pos": "WH"}, 'UH':{"pos": "INT"}}

# tagger_dataset = pd.read_csv('dataset/tagger/Indonesian_Manually_Tagged_Corpus_ID.tsv', sep="\t", names=["word", "tag"])
# tags = tagger_dataset.tag.value_counts()
# taguh = tagger_dataset[tagger_dataset.tag == 'UH']

    def preprocess_tagger(self):
        #tagger.tag.unique() # except NaN/nan

        #TRAIN_DATA = [
        #    ("I like green eggs", {"tags": ["N", "V", "J", "N"]}),
        #    ("Eat blue ham", {"tags": ["V", "J", "N"]}),
        #]

        sentence_tmp = ''
        tags_tmp = []
        data_train = []
        for _, _df in self.tagger_dataset.iterrows():
            
            if re.search(r'\<\bkalimat\b id=.+\>', _df.word): # <kalimat id=256a>
                continue
            if not re.match(r'\</\bkalimat\b\>', _df.word): # </kalimat>
                sentence_tmp += ' ' + _df.word
                tags_tmp.append(_df.tag.upper())
            else:
                if re.search(r'\t', sentence_tmp):
                    sentence_tmp = ''
                    tags_tmp = []
                    continue
                data_train.append((sentence_tmp.strip(), {"tags": tags_tmp}))
                sentence_tmp = ''
                tags_tmp = []
        
        return data_train
    
<<<<<<< HEAD
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
    
=======
>>>>>>> 741318f9d71e5903382ec7bf3b3c5e157bef6995



