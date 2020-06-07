train = [('I love this sandwich.', 'pos'),
        ('this is an amazing place!', 'pos'),
        ('I feel very good about these beers.', 'pos'),
        ('this is my best work.', 'pos'),
        ("what an awesome view", 'pos'),
        ('I do not like this restaurant', 'neg'),
        ('I am tired of this stuff.', 'neg'),
        ("I can't deal with this", 'neg'),
        ('he is my sworn enemy!', 'neg'),
        ('my boss is horrible.', 'neg')]

test = [('the beer was good.', 'pos'),
        ('I do not enjoy my job', 'neg'),
        ("I ain't feeling dandy today.", 'neg'),
        ("I feel amazing!", 'pos'),
        ('Gary is a friend of mine.', 'pos'),
        ("I can't believe I'm doing this.", 'neg')]


from textblob.classifiers import NaiveBayesClassifier
cl = NaiveBayesClassifier(train)
print("Using NaiveBayesClassifier")
#Call the classify(text) method to use the classifier.
print("Classifier of the sentence 'This is an amazing library!'")
print(cl.classify("This is an amazing library!"))
#You can get the label probability distribution with the prob_classify(text) method.
print(" you find the  probability distribution on 'This one's a doozy.'")
prob_dist = cl.prob_classify("This one's a doozy.")
print(prob_dist.max())
print("probability distribution probability of Positive and Negative")
print(round(prob_dist.prob("pos"), 2))
print(round(prob_dist.prob("neg"), 2))
print("After Using TextBlob")
from textblob import TextBlob
print("Sentence is 'The beer is good. But the hangover is horrible.'")
blob = TextBlob("The beer is good. But the hangover is horrible.", classifier=cl)
print("Classifier of the sentence",blob.classify())
for s in blob.sentences:
    print(s)
    print(s.classify())
print("Accuracy is ",cl.accuracy(test))
cl.show_informative_features(5)

new_data = [('She is my best friend.', 'pos'),
    ("I'm happy to have a new friend.", 'pos'),
    ("Stay thirsty, my friend.", 'pos'),
    ("He ain't from around here.", 'neg')]
#Use the update(new_data) method to update a classifier with new training data.
cl.update(new_data)
print("After update the training data Accurarcy is ",cl.accuracy(test))
def end_word_extractor(document):
    tokens = document.split()
    first_word, last_word = tokens[0], tokens[-1]
    feats = {}
    feats["first({0})".format(first_word)] = True
    feats["last({0})".format(last_word)] = False
    return feats
features = end_word_extractor("I feel happy")
assert features == {'last(happy)': False, 'first(I)': True}
#here we enter the feature_extractor
cl2 = NaiveBayesClassifier(test, feature_extractor=end_word_extractor)
blob = TextBlob("I'm excited to try my new classifier.", classifier=cl2)
print("Classifier of the sentence After enter the feature Extractor",blob.classify())
