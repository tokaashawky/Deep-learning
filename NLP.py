import nltk 
import re 
from nltk.stem import PorterStemmer
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('twitter_samples')

stop_words=stopwords.words('english')
print(stop_words)

positive_tweets=twitter_samples.strings('positive_tweets.json')
negative_tweets=twitter_samples.strings('negative_tweets.json')
print(len(positive_tweets))

print(positive_tweets)

print(len(negative_tweets))


def cleantext(tweet):
    tweet=re.sub('(@|#)\w*',"",tweet)
    tweet=re.sub('http?:\/\/\S+'," ",tweet)
    tweet=re.sub('^\s+',"",tweet)
    tweet=re.sub('\s+$',"",tweet)
    return tweet


def process_sentances (tweets):
    clean_tweet=[]
    for tweet in tweets:
        tweet=clean_text(tweet)
        tweet =tweet.spilt()
        c_tweet = [word.lower() for word in tweet if word.lower() not in stop_words]  
        
        ps= PorterStemmer()
        clean_t=[ps.stem(word) for word in c_tweet ]
        clean_tweet.append(clean_t)
        return clean_tweet
    






