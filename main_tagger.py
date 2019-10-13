import os, sys, spacy

from preprocess import Preprocess

DIR_this = os.path.dirname(os.path.abspath(__file__))
DIR_model = os.path.dirname(os.path.join(DIR_this, 'model', 'tagger'))

class MainTagger(object):
    self.preprocess = Preprocess()
    self.TRAIN_DATA_TAGGER = self.preprocess.preprocess_tagger()
    self.OUTPUT_DIR = os.path.join(DIR_model)
    self.TAG_LABEL = {'NN':{"pos": "NOUN"}, 'SC':{"pos": "ADP"}, 'VB':{"pos": "VERB"},\
                          'NNP':{"pos": "PROPN"}, 'JJ':{"pos": "ADJ"}, 'RB':{"pos": "ADV"},\
                          'IN':{"pos": "PRP"}, 'Z':{"pos": "PUNCT"}, 'CD':{"pos": "SCONJ"},\
                          'CC':{"pos": "CONJ"}, 'PR':{"pos": "PROPN"}, 'PRP':{"pos": "PART"},\
                          'MD':{"pos": "AUX"}, 'FW':{"pos": "FOREIGN"}, 'NEG':{"pos": "NEG"},\
                          'DT':{"pos": "DET"}, 'NND':{"pos": "NN"}, 'SYM':{"pos": "SYM"},\
                          'RP':{"pos": "NUM"}, 'OD':{"pos": "OD"},'X':{"pos": "UNK"},\
                          'WH':{"pos": "WH"}, 'UH':{"pos": "INT"}}
    self.epochs = 50
    self.lang = 'id'
    if not train:
        self.model = self.load_model()
    else:
        self.model = spacy.blank(self.lang)


    def main(self):
        """Create a new model, set up the pipeline and train the tagger. In order to
        train the tagger with a custom tag map, we're creating a new Language
        instance with a custom vocab.
        """
        
        # add the tagger to the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        tagger = self.model.create_pipe("tagger")
        # Add the tags. This needs to be done before you start training.
        for tag, values in self.TAG_LABEL.items():
            tagger.add_label(tag, values)
        
        nlp.add_pipe(tagger)
        
        optimizer = self.model.begin_training()
        for i in range(self.epochs):
            random.shuffle(self.TRAIN_DATA_TAGGER)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(self.TRAIN_DATA_TAGGER, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                self.model.update(texts, annotations, sgd=optimizer, losses=losses)
            print("Losses", losses)

        # save model to output directory
        if self.OUTPUT_DIR is not None:
            output_dir = Path(self.OUTPUT_DIR)
            if not self.OUTPUT_DIR.exists():
                self.OUTPUT_DIR.mkdir()
            self.model.to_disk(self.OUTPUT_DIR)
            print("Saved model to", self.OUTPUT_DIR)
    
    def load_model(self):
        model = spacy.load(self.OUTPUT_DIR)
        return model

    def predict(self, text):
        # test the trained model
        doc = self.load_model(text)
        return [(t.text, t.tag_, t.pos_) for t in doc]


if __name__ == "__main__":
    obj = MainTagger()
    obj.main_tagger()