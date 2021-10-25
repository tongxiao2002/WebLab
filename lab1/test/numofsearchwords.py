import nltk
import gensim
import tqdm
import json 

s = ""

n1 = 57802
n2 = 64592 + n1
n3 = 57456 +n2
n4 = 63245+n3
n5 = 63147+n4

a =1
num = int(input())
linenum = 0 
if num < n1:
    a =1
    linenum = num
elif num <n2:
    a = 2
    linenum = num - n1
elif num <n3:
    a =3
    linenum = num - n2
elif num<n4:
    a = 4
    linenum = num - n3
else:
    a = 5
    linenum = num - n4

filestr = "lab1/data/" + "2018_0" + str(a) + ".json"

with open(filestr, "r") as fin:
    count = -1
    for line in fin:
        count+=1
        if count == linenum:
            data = json.loads(line)
            s = data["text"]
            sid = data["id"]
            print(sid)
            break


search_words = []
counter = 0
counter1 = 0
porter_stemmer = nltk.stem.PorterStemmer()
with open("lab1/data/searchwords.txt", "r", encoding="utf-8") as fin:
    for line in fin:
        search_words.append(porter_stemmer.stem(line.strip()))

word_list = [porter_stemmer.stem(i) for i in (gensim.utils.tokenize(s, lowercase=True, deacc=True))]

# tokenize

for word in search_words:
    if word in word_list:
        counter+=1
    
for word in word_list:
    if word in search_words:
        counter1+=1

print(counter)
print(counter1)
print(len(word_list))
