import time
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from random import randint
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


# receives a tweet string & returns an array of tokens
def tokenzier(tweet):
    filtered_url = re.sub(r"http\S+", repl="", string=tweet)
    tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
    tokenized_tweet = tweet_tokenizer.tokenize(filtered_url)
    return tokenized_tweet


# receives array of tokens & returns them as normalized tokens
def normalizer(tokenized_tweet):
    # filter out words with numbers and punctuations
    filtered_nonalphabet = [word for word in tokenized_tweet if word.isalpha() and len(word) > 1]
    stop_words = set(stopwords.words('english'))
    stop_words.add("ude")  # unnecessary word which appears in almost all tweets
    filtered_stopwords = [word for word in filtered_nonalphabet if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word, "v") for word in filtered_stopwords]  # v is for verb - part of speech tag
    return lemmatized


# receives tweet string & returns a string which contains normalized terms, separated by space
def tokenize_normalize(tweet):
    tokenized = tokenzier(tweet)
    normalized = normalizer(tokenized)
    space = " "
    sentence = space.join(normalized)
    return sentence


# receives array of tokenized_normalized tweets [tweet1, tweet2...]
def sentiment_counter(tweets):
    sentiment_count = {"Positive": 0, "Neutral": 0, "Negative": 0}
    sid = SentimentIntensityAnalyzer()
    for tweet in tweets:
        score_per_sentence = sid.polarity_scores(tweet)  # example: {compound:0.8316, neg:0.0, neu:0.254, pos:0.746}
        compound_score = score_per_sentence["compound"]
        # scoring scheme as recommended by vader developers
        if compound_score >= 0.05:
            sentiment_count["Positive"] += 1
        elif -0.05 < compound_score < 0.05:
            sentiment_count["Neutral"] += 1
        elif compound_score <= -0.05:
            sentiment_count["Negative"] += 1
    return sentiment_count


def sentiment_analysis_graph(sentiment_counter, location=None):
    r = lambda: randint(0, 255)
    color = ('#%02X%02X%02X' % (r(), r(), r()))  # int to hex, if less than 2 digits, 0 is appended
    bars = plt.bar(list(sentiment_counter.keys()), list(sentiment_counter.values()),
                   color=color)
    plt.title("Game of Thrones - Total Tweets Sentiment Analysis" if location is None else str(location + " Analysis"))
    plt.xlabel("Sentiment", fontsize=14)
    plt.ylabel("Total Tweets", fontsize=14)
    # plot numbers on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.3, yval + 0.03, yval, fontsize=14)
    plt.show()


def generate_word_cloud(total_tweets, location=None):
    corpus_frequency = {}
    for tweet in total_tweets:
        split_tokens = tweet.split()
        for token in split_tokens:
            corpus_frequency[token] = corpus_frequency.get(token, 0) + 1
    wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stopwords.words("english"),
                          min_font_size=10).generate_from_frequencies(corpus_frequency)
    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.title("Game of Thrones WordCloud" if location is None else str(location + " WordCloud"), fontsize=14)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


# -----------------------------MAIN---------------------
if __name__ == "__main__":
    start = time.time()
    excel_data = pd.read_excel('AlonGOT_7_D_before_SA-500000.xlsx', nrows=100, usecols="F,G")

    # Sentiment Analysis and wordCloud for all tweets
    all_tweets = excel_data["Text"]
    tokenized_normalized_tweets = [tokenize_normalize(tweet) for tweet in all_tweets]
    sentiment_count = sentiment_counter(tokenized_normalized_tweets)
    sentiment_analysis_graph(sentiment_count)
    generate_word_cloud(tokenized_normalized_tweets)

    # Sentiment Analysis and wordcloud per location
    top3_twitting_locations = Counter(excel_data["Geo Location"]).most_common()[1:4]
    top3_location_names = [item[0] for item in top3_twitting_locations]
    for location in top3_location_names:
        excel_tweets_with_location = excel_data.loc[excel_data["Geo Location"] == location, ["Text"]]
        tweets_per_location = excel_tweets_with_location["Text"]
        tokenized_normalized_tweets = [tokenize_normalize(tweet) for tweet in tweets_per_location]
        sentiment_count = sentiment_counter(tokenized_normalized_tweets)
        sentiment_analysis_graph(sentiment_count, location)
        generate_word_cloud(tokenized_normalized_tweets, location)

    print("Time of execution: %d" % (time.time() - start))
