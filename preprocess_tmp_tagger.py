import os, sys
import spacy
import numpy as np
import pandas as pd

import re


THIS_DIR = os.getcwd()
DIR_TAGGER = os.path.join(THIS_DIR, "dataset/tagger/Indonesian_Manually_Tagged_Corpus_ID.tsv")



tagger = pd.read_csv(DIR_TAGGER, sep="\t", names=["word", "tag"])

#tagger = tagger.dropna(subset=["tag"])
tagger.tag.unique() # except NaN/nan
TAG_MAP = {'NN':{"pos": "NOUN"}, 'SC':{"pos": "ADP"}, 'VB':{"pos": "VERB"}, 'NNP':{"pos": "PROPN"}, 'JJ':{"pos": "ADJ"}, 'RB', 'IN', 'Z':{"pos": "PUNCT"}, 'CD':{"pos": "SCONJ"}, 'CC':{"pos": "CONJ"},
           'PR', 'PRP':{"pos": "PART"}, 'MD', 'FW', 'NEG', 'DT':{"pos": "DET"}, 'NND', 'SYM':{"pos": "SYM"}, 'RP':{"pos": "NUM"}, 'OD',
           'X':{"pos": "X"}, 'WH', 'UH'}

#TRAIN_DATA = [
#    ("I like green eggs", {"tags": ["N", "V", "J", "N"]}),
#    ("Eat blue ham", {"tags": ["V", "J", "N"]}),
#]

sentence_tmp = ''
tags_tmp = []
data_train = []
for _, _df in tagger.iterrows():
    
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
    

def main(lang="en", output_dir=None, n_iter=25):
    """Create a new model, set up the pipeline and train the tagger. In order to
    train the tagger with a custom tag map, we're creating a new Language
    instance with a custom vocab.
    """
    nlp = spacy.blank(lang)
    # add the tagger to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    tagger = nlp.create_pipe("tagger")
    # Add the tags. This needs to be done before you start training.
    for tag, values in TAG_MAP.items():
        tagger.add_label(tag, values)
    nlp.add_pipe(tagger)
    
    optimizer = nlp.begin_training()
    for i in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, losses=losses)
        print("Losses", losses)

    # test the trained model
    test_text = "I like blue eggs"
    doc = nlp(test_text)
    print("Tags", [(t.text, t.tag_, t.pos_) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the save model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc = nlp2(test_text)
        print("Tags", [(t.text, t.tag_, t.pos_) for t in doc])


if __name__ == "__main__":
    plac.call(main)

