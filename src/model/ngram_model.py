import os
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
        for word, count in dictionary.items():
            if count < int(os.getenv("UNK_THRESHOLD")):
                dictionary[word] = "UNK"
                del dictionary[count]
        return dictionary
    


#testing the NGramModel class
m = NGramModel()
print(m.build_vocab("tests/save_output.txt"))