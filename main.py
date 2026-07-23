
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor
from src.evaluation.evaluator import Evaluator
import argparse
import os
import logging

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=str)
args = parser.parse_args()

if args.step == "dataprep" or args.step == "all":
    n = Normalizer()
    a = n.load("data/raw/train")
    final = []
    for book in a:
        b = n.sentence_tokenize(book)
        for c in b: 
            x = n.normalize(c)
            final.append(n.word_tokenize(x))
    n.save(final, "data/processed/train_tokens.txt")

if args.step == "model" or args.step == "all":
    m = NGramModel()
    m.build_vocab("data/processed/train_tokens.txt")
    count, probability = m.build_counts_and_probabilities("data/processed/train_tokens.txt")
    m.save_model("data/model/model.json")
    m.save_vocab("data/model/vocab.json")

if args.step == "inference" or args.step == "all":
    m = NGramModel()
    m.load("data/model/model.json", "data/model/vocab.json")
    n = Normalizer()
    p = Predictor(m, n)

    while True:
        text = input()
        if text == "quit":
            break
        try:
            top = int(os.getenv("TOP_K"))
            predictions = p.predict_next(text,top)
            print(predictions)
        except KeyError:
            logger.error(f"Missing config variable: {top}. Check config/.env.")

if args.step == "evaluate" or args.step == "all":
    m = NGramModel()
    m.load(os.getenv("MODEL"), os.getenv("VOCAB"))
    n = Normalizer()
    e = Evaluator(m, n)
    e.run(os.getenv("EVAL_TOKENS"))


#python main.py --step evaluate