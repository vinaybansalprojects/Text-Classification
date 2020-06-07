import spacy
nlp=spacy.load('en_core_web_sm')
data=nlp("Spacy is a Amazing tOOl")
myfile=open("1.Abstract.txt").read()
doc_file=nlp(myfile)
#print(doc_file)
#Tokenization
'''for num,sentence in enumerate(doc_file.sents):
    print(f'{num}:{sentence}')
'''
#Word Token
'''for token in doc_file:
    print(token.text,token.shape_,token.is_alpha,token.is_stop,token.pos_,token.tag_,token.dep_,token.lemma_)
'''
'''[print(token.text) for token in data]
print(token.text.split(" "))
'''
'''
for token in doc_file.ents:
    print(token.text,token.label_)
'''
'''from spacy import displacy
displacy.serve(doc_file,style='dep')
'''

from spacy.lang.en.stop_words import STOP_WORDS
print(len(STOP_WORDS))
'''
for word in doc_file:
    if word.is_stop == False:
        print(word)
'''

mylist=[word for word in doc_file if word.is_stop == False]
print(mylist)

'''
def mycustom_boundary(docx):
    for token in docx[:-1]:
        if token.text == '.':
            docx[token.i+1].is_sent_start=True
    return docx
nlp.add_pipe(mycustom_boundary,before='parser')
for num,sentence in enumerate(doc_file.sents):
    print(f'{num}:{sentence}')
'''
