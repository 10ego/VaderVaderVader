from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

analyzer = SentimentIntensityAnalyzer()
is_neu_p = 0
is_neu_n = 0
pos_count = 0
is_pos = 0
neg_count = 0
is_neg = 0
threshold = 0.05
pos_subj = 0
neg_subj = 0
neu_subj = 0

with open('positive.txt','r') as f:
    for line in f.read().split('\n'):
        subj = TextBlob(line)
        score = analyzer.polarity_scores(line)
        #if score['neu'] > score['pos']:
        if score['compound'] < threshold and score['compound'] > -threshold:
            is_neu_p += 1
            if subj.sentiment.subjectivity >= 0.5:
                pos_subj+=1
        else:
            if not score['neg'] > threshold:
                if score['pos']-score['neg'] > 0:
                    is_pos +=1
                    if subj.sentiment.subjectivity >= 0.5:
                        pos_subj+=1
                pos_count +=1
            
            

with open('negative.txt','r') as f:
    for line in f.read().split('\n'):
        subj = TextBlob(line)
        score = analyzer.polarity_scores(line)
        #if score['neu'] > score['neg']:
        if score['compound'] < threshold and score['compound'] > -threshold:
            is_neu_n += 1
            if subj.sentiment.subjectivity >= 0.5:
                neg_subj+=1
        else:
            if not score['pos'] > threshold:
                if score['neg']-score['pos'] > 0:
                    is_neg +=1
                    if subj.sentiment.subjectivity >= 0.5:
                        neg_subj+=1
                neg_count +=1
        

print("Positive accuracy = {}% via {} samples".format(is_pos/pos_count*100, pos_count))
print("Negative accuracy = {}% via {} samples".format(is_neg/neg_count*100, neg_count))
print("Total of {} positive messages are subjective".format(pos_subj))
print("Total of {} negative messages are subjective".format(neg_subj))
print("{} positive messages are actually neutral".format(is_neu_p))
print("{} negative messages are actually neutral".format(is_neu_n))
