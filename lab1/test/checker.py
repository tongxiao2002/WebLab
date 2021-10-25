import nltk
import tqdm

words = []
search_words = []
with open("words.txt", "r", encoding="utf-8") as fin:
    for line in fin:
        words.append(line.strip())

porter_stemmer = nltk.stem.PorterStemmer()
with open("lab1/data/searchwords.txt", "r", encoding="utf-8") as fin:
    for line in fin:
        search_words.append(porter_stemmer.stem(line.strip()))

num = 0
for search_word in tqdm.tqdm(search_words):
    if search_word not in words:
        num += 1
        print(search_word)
print(num)