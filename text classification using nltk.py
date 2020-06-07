import io 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
#word_tokenize accepts a string as an input, not a file. 
stop_words = set(stopwords.words('english')) 
file1 = open("1.Abstract.txt") 
line = file1.read()# Use this to read file content as a stream: 
words = line.split() 
for r in words: 
	if not r in stop_words: 
		appendFile = open('filteredtext.txt','a') 
		appendFile.write(" "+r) 
		appendFile.close() 
file2=open("filteredtext.txt")
line=file2.read()
print(line.split())
