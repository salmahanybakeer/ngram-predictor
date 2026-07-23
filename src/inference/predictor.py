from dotenv import load_dotenv
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
load_dotenv("config/.env")
import os
class Predictor:
    """
    accepting a pre-loaded NGramModel and Normalizer via the constructor, 
    normalizing input text, 
    and returning the top-k predicted next words sorted by probability.
    """
    def __init__(self, model, normalizer):
        """
        function: accept a pre-loaded NGramModel and Normalizer instance.
        parameters: model , normlaizer 
        return: none
        """
        self.model = model
        self.normalizer = normalizer
    def normalize(self,text):
        """
        function: call Normalizer.normalize(text); extract last NGRAM_ORDER − 1 words as context
        parameters: text
        return: the last NGRAM_ORDER - 1 words as a list.
        """
        x = self.normalizer.normalize(text).split()
        num = int(os.getenv("NGRAM_ORDER")) - 1
        return x[-num:]
    def map_oov(self,context):
        """
        function: replace out-of-vocabulary words with <UNK>
        parameters: context
        return: list with out-of-vocabulary words rrplaced by <UNK> 
        """ 
        out = []
        for i in context:
            if i in self.model.vocab:
                out.append(i)
            else:
                out.append("<UNK>")
        return out
    def predict_next(self,text, k):
        """
        function: orchestrate normalize → map_oov → NGramModel.lookup() → return top-k words sorted by probability
        parameters: text, k
        return: list of top-k predicted next words, sorted by probability (highest first)
        """
        if text == "":
            print("Input text is empty. Please type at least one word.")
        else:
            r = self.normalize(text)
            s = self.map_oov(r) 
            st = " ".join(s) 
            result = self.model.lookup(st)
            sorted_list = sorted(result,key=result.get,reverse = True)
            out = sorted_list[:k]
            return out 
       
            





"""
#TESTING
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
m = NGramModel()
m.load("data/model/modell.json", "data/model/vocab.json")   # or your test files
n = Normalizer()
p = Predictor(m, n)
print(p.normalize("Holmes looked at Watson very carefully."))
print(p.map_oov(["at", "watson", "xyzzy"]))

m = NGramModel()
m.load("tests/modell.json", "tests/save_output.json")
n = Normalizer()
p = Predictor(m, n)
print(p.normalize("Holmes looked at Watson very carefully."))
print(p.map_oov(["watson", "holmes", "xyzzy"]))
print(p.predict_next("Holmes looked at", 3))
"""
