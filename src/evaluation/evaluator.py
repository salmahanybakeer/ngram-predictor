import math
import os
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel

class Evaluator:
    """
    Computes perplexity of a pre-loaded NGramModel on a held-out
    evaluation corpus, using log2 probabilities from NGramModel.lookup()
    with backoff.
    """
    def __init__(self,model, normalizer):
        """
        function: accept a pre-loaded NGramModel and Normalizer instance
        parameters: model , normalizer
        return: nothing
        """
        self.model = model
        self.normalizer = normalizer
    def score_word(self,word, context):
        """
        function: Return log₂ P(word | context) via NGramModel.lookup(); return None if zero probability at all orders
        parameters: word , context
        return: log₂ P(word | context) or none (if zero probability)
        """
        x = self.model.lookup(context)
        probability = x.get(word, 0)
        if probability == 0:
            return None
        else:
            return math.log2(probability)
    def compute_perplexity(self,eval_file):
        """
        function: compute perplexity over the full eval corpus; print a warning if more than 20% of words are skipped
        parameters: eval_file
        return: perplexity
        """
        context_size = int(os.getenv("NGRAM_ORDER")) - 1
        total_log_prob = 0
        n_words = 0
        n_skipped = 0
        for line in open(eval_file, "r", encoding="utf-8"):
            words = line.split()
            for i in range(len(words)):
                word = words[i]
                context_words = words[max(0, i-context_size):i]
                context = " ".join(context_words)
                score = self.score_word(word, context)
                if score is None:
                    n_skipped += 1
                else:
                    total_log_prob += score
                    n_words += 1
        H = -(1/n_words) * total_log_prob
        perplexity = 2 ** H
        if n_skipped / (n_words + n_skipped) > 0.20:
            print(f"Warning: {n_skipped} words skipped (more than 20% of total).")
        return perplexity, n_words, n_skipped
    def run(self,eval_file):
        """
        function: orchestrate compute_perplexity 
        prameters: eval_file
        return:  prints result
        """
        perp,ev,skip = self.compute_perplexity(eval_file)
        print(f"Perplexity: {perp}")
        print(f"Words evaluated: {ev}")
        print(f"Words skipped (zero probability): {skip}")
"""
#TESTS
m = NGramModel()
m.load("data/model/model.json", "data/model/vocab.json")
n = Normalizer()
e = Evaluator(m, n)

#print(e.compute_perplexity("EVAL_TOKENS"))
e.run("EVAL_TOKENS")
"""