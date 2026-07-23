from email.mime import text
from json import load
import string
import os
import logging
logger = logging.getLogger(__name__)

class Normalizer:
    """ 
    loads, cleans, tokenizes, and saves the corpus
    """
    def lowercase(self, text):
        """ 
        function: lowercases the text
        parameters: text
        return: lowercased text
        """
        return text.lower()
    def remove_punctuation(self, text):
        """ 
        function: removes punctuation from the text
        parameters: text
        return: text without punctuation
        """
        str = ""
        quotations = ["”", "“", "‘", "’"]
        for char in text:
            if char not in string.punctuation and char not in quotations:
                str += char
        return str
    def remove_numbers(self, text):
        """ 
        function: removes numbers from the text
        parameters: text
        return: text without numbers
        """
        str = ""
        for char in text:
            if not(char >='0' and char<='9'):
                 str += char
        return str
    def remove_whitespaces(self, text):
        """ 
        function: removes whitespaces from the text
        parameters: text
        return: text without whitespaces
        """
        new_text = text.split()
        return " ".join(new_text)
    def strip_gutenberg(self,text):
        """ 
        function: removes header and footer from the text
        parameters: text
        return: text without header and footer
        """
        start = text.find("START OF THE PROJECT GUTENBERG EBOOK")
        end =text.find("END OF THE PROJECT GUTENBERG EBOOK")
        if start !=-1 and end !=-1:
            start_line_end = text.find("\n", start)
            return text[start_line_end + 1 : end-4]
        else:
            return text
    def load(self,folder_path):
        """ 
        function: loads the text files from the folder
        parameters: folder_path
        return: list of texts
        """
        text = []
        try:
           for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    with open(os.path.join(folder_path, filename), encoding="utf-8") as f:
                        text.append(f.read())
        except FileNotFoundError:
                    logger.error(f"Folder not found: {folder_path}. Check TRAIN_RAW_DIR in config/.env.")
        return text 
    def normalize(self,text):
        """ 
        function: apply all the normalization functions to the text
        parameters: text
        return: normalized text
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespaces(text)
        return text
    #done before calling the normalize method
    def sentence_tokenize(self,text):
        """ 
        function: split text into a list of sentences
        parameters: text
        return: list of sentences
        """
        ret = []
        str = ""
        t = [".","!","?"]
        for p in text:
            if p in t:
                ret.append(str + p)
                str = ""
            else:
                str += p     
        return ret
    def word_tokenize(self,sentence):
        """ 
        function: split sentence into a list of words
        parameters: sentence
        return: list of words
        """
        return sentence.split()
    def save(self,sentences, filepath):
        """ 
        function: write tokenized sentences to output file
        parameters: sentences, filepath
        return: nothing
        """
        with open(filepath, "w", encoding="utf-8") as f:
            for s in sentences:
                f.write(' '.join(s) + "\n")


"""
#testing the class
n = Normalizer()

#lowercase done
print(n.lowercase("HELLO WORLD"))

#remove punctuation done
print(n.remove_punctuation("Hello, World!."))

#remove numbers done
print(n.remove_numbers("Hello 123 World!"))

#remove whitespaces done
print(n.remove_whitespaces("Hello    World!\nHi Salma"))

#remove header and footer done
with open("tests/guternberg_test.txt", encoding="utf-8") as f:
    raw_text = f.read()
result = n.strip_gutenberg(raw_text)
print(result)

#load method done
text = n.load("data/raw/train/")
print(text)

#normalize method done
text = n.normalize("Hello, World! 123")
print(text)

#sentence tokenize method done
sentences = n.sentence_tokenize("Hello, World! How are you? I am fine.")
print(sentences)

#word tokenize method done
words = n.word_tokenize("Hi I am Salma")
print(words)

#save method done
test_sentences = [["the", "adventure", "began"], ["holmes", "examined", "the", "letter"]]
n.save(test_sentences, "tests/save_output.txt")
"""