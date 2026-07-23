from itertools import count
from multiprocessing import context
import os
from dotenv import load_dotenv
from nltk import probability
import json
load_dotenv("config/.env")
import logging
logger = logging.getLogger(__name__)

class NGramModel:
    def build_vocab(self,token_file):
        """
        function: build vocabulary from tokenized sentences
        parameters: token_file
        return: vocabulary dictionary
        """
        dictionary = {}
        for line in open(token_file ,"r", encoding="utf-8"):
            for word in line.split():
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1
        
        unk_count = 0 
        removed_words = []  
        for word, count in dictionary.items():
            if count < int(os.getenv("UNK_THRESHOLD")):
                unk_count += count
                removed_words.append(word)
    
        for removed_word in removed_words:  
            del dictionary[removed_word]
        dictionary["UNK"]  = unk_count
        self.vocab = dictionary
        return dictionary
    def build_counts_and_probabilities(self,token_file):
        """
        function: build counts and probabilities from tokenized sentences
        parameters: token_file
        return: counts and probabilities dictionaries
        """
        n=int(os.getenv("NGRAM_ORDER"))
        count = {}
        for r in range(1,n+1):
            count[f"{r}gram"] = {}
            for line in open(token_file, "r", encoding="utf-8"):
                words = []
                for word in line.split():
                    words.append(word)
                for i in range(len(words)-r+1):
                    x = words[i:i+r]
                    out = " ".join(x)
                    count[f"{r}gram"][out] = count[f"{r}gram"].get(out, 0) + 1
        probability = {}
        for r in range(1,n+1):
            probability[f"{r}gram"] = {}
            enabled = os.getenv("SMOOTHING")
            for out in count[f"{r}gram"]:
                 k = float(os.getenv("SMOOTHING_K"))
                 if r==1:
                    if enabled:
                        probability[f"{r}gram"][out] = (count[f"{r}gram"][out]+k) / (sum(count[f"{r}gram"].values())+k*len(self.vocab))
                    else:
                        probability[f"{r}gram"][out] = count[f"{r}gram"][out] / sum(count[f"{r}gram"].values())
                 else:
                    x = out.split()[:-1]   
                    context = " ".join(x)            
                    last_word = out.split()[-1]       
                    if context not in probability[f"{r}gram"]:   
                        probability[f"{r}gram"][context] = {}    
                    if enabled: 
                        probability[f"{r}gram"][context][last_word] = (count[f"{r}gram"][out]+k) / (count[f"{r-1}gram"][context]+k*len(self.vocab))
                    else:
                         probability[f"{r}gram"][context][last_word] = count[f"{r}gram"][out] / count[f"{r-1}gram"][context]
        self.count = count
        self.probability = probability
        return count, probability              
    def lookup(self, context):
        """
        function: finds the word with the highest probability to come after the given context
        parameters: context
        return: words that may com eafter contextwith the their probability
        """
        n=int(os.getenv("NGRAM_ORDER"))
        for r in range(n,1,-1):
            if context in self.probability[f"{r}gram"]:
                return self.probability[f"{r}gram"][context]
        return  self.probability["1gram"]
    def save_model(self, model_path):
        """
        function: save all probability tables to model.json
        parameters: model_path
        return: nothing
        """
        with open(model_path, "w" , encoding="utf-8") as f:
            json.dump(self.probability, f)
    def save_vocab(self, vocab_path):
        """
        function: save vocabulary list to vocab.json
        parameters: vocab_path
        return: nothing
        """
        with open(vocab_path, "w" , encoding="utf-8") as f:
            json.dump(list(self.vocab.keys()), f)
    def load(self,model_path, vocab_path):
        """
        function: load model and vocab from json files
        parameters: model_path, vocab_path
        return: model and vocab
        """
        try:
            with open(model_path, "r" , encoding="utf-8") as f:
                probability = json.load(f)
            with open(vocab_path, "r" , encoding="utf-8") as f:
                vocab = json.load(f)
            self.probability = probability
            self.vocab = vocab
        except FileNotFoundError:
            logger.error("model.json not found. Run the Model module first.")
        except json.JSONDecodeError:
            logger.error("model.json is malformed. Re-run the Model module.")
        return self.probability, self.vocab


"""
#testing the NGramModel class
m = NGramModel()

#build_vocab
print(m.build_vocab("tests/save_output.txt"))

#build_counts_and_probabilities
count, probability = m.build_counts_and_probabilities("tests/save_output.txt")
print(probability["2gram"])
print(probability["1gram"])

#lookup
print(m.lookup("said"))

#save_model
m.save_model("tests/modell.json")

#save_vocab
m.save_vocab("tests/save_output.json")

#load
m.load( "tests/modell.json","tests/save_output.json")
print(m.lookup("watson"))
print(m.vocab)



#Milestone 2 tests
m2 = NGramModel()
print(m2.build_vocab("tests/m2_test.txt"))
count, probability = m2.build_counts_and_probabilities("tests/m2_test.txt")
print(probability["1gram"]["the"])
print(probability["2gram"]["the"])
m2.save_model("tests/modell.json")
m2.save_vocab("tests/save_output.json")
"""