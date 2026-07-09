from src.data_prep.normalizer import Normalizer

n = Normalizer()
a = n.load("data/raw/train/")
final = []
for book in a:
    b = n.sentence_tokenize(book)
    for c in b: 
        x = n.normalize(c)
        final.append(n.word_tokenize(x))
n.save(final, "data/processed/train_tokens.txt")