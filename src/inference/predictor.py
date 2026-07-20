from dotenv import load_dotenv
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
    
"""
#TESTING
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
m = NGramModel()
m.load("data/model/modell.json", "data/model/vocab.json")   # or your test files
n = Normalizer()
p = Predictor(m, n)
print(p.normalize("Holmes looked at Watson very carefully."))
"""