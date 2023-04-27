# main.py
from datetime import datetime
from http.client import BadStatusLine
from urllib.error import URLError

import twitter
import json
import csv
import os
import re
import sys
import time
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
#from spacytextblob.spacytextblob import TextBlob
# from textblob import TextBlob

#import config

from TwitterCookbook import oauth_login, make_twitter_request, harvest_user_timeline, twitter_search

# ----------------------------------------------------------------------------------------------------------------------
# Player Object Creation - Used throughout the project
# Keeps track of player name, any aliases (tbd), stats and sentiment score to be used to determine
# likelihood of winning
# ----------------------------------------------------------------------------------------------------------------------
class Player:
    

    #making a user instance
    def __init__(self, name, aliases, stats, polScore):
        self.name = name
        self.aliases = aliases
        self.stats = stats
        self.pScore = polScore

    #user methods----------------------------------------------------------------------------
    def checkMention(self, ls, text): #determine if a player is mentioned by name or alias
        if self.name in ls:
            return True
        #check aliases
        #for alias in text:
         #if alias in text:
                #return True
        #return False
    
    #def getStats(self, str): #retrieve stats from documents if not initally given


if __name__ == '__main__':
    twitter_api = oauth_login()

    print("Twitter Api {0}\n".format(twitter_api))

    #dictionary of nominees and starting score (0) {Player Name : 0}
    nomList = {"Joel Embiid": 0,
               "Giannis": 0,
               "Nikola Jokic": 0}

    # ------------------------------------------------------------------------------------------------------------------
    # Task 2
    # Use Twitter's API to collect tweets from fans about NBA players' performances throughout the regular
    # season. Use relevant hashtags or keywords to filter tweets specifically about NBA players.
    # ------------------------------------------------------------------------------------------------------------------

    tweets_folder_name = 'Tag-Filtered-Tweets'
    os.makedirs(tweets_folder_name, exist_ok=True)


    mvp_voter_path = os.path.join(os.path.dirname(__file__), 'NBA-Season-Award-Voters/MVP Voters By Year.csv')

    #TAGS OF INTEREST IN TWEET RETRIEVAL
    tags = ["jokic", "giannis", "embiid"]

    # Harvest tweets from each voter
    for t in tags:
        tweets = twitter_search(twitter_api, t, max_results=10)

        print("{0} tweets: ".format(t))
        print(tweets)

        # Filter tweets by date and possibly player names
        filtered_tweets = []
        for tweet in tweets:
            # print("Tweet created at {0}\n".format(created_at))
            filtered_tweets.append(tweet)

        print("{0} filtered tweets: ".format(t))
        print(filtered_tweets)

        # Save the filtered tweets to a json file within Voter-Filtered-Tweets
        filename = f'{t}_filtered_tweets.json'
        filepath = os.path.join(tweets_folder_name, filename)
        with open(filepath, 'w') as f:
            json.dump(filtered_tweets, f, indent="")
            print("{0} tweets dumped in {1}".format(t, filename))

    # ------------------------------------------------------------------------------------------------------------------
    # Task 3
    # Analyze the sentiment of the tweets using natural language processing (NLP) techniques. NLP libraries: NLTK or
    # spaCy to identify the sentiment of the tweets (positive, negative, or neutral) towards specific players.
    # https://pypi.org/project/spacytextblob/
    # ------------------------------------------------------------------------------------------------------------------

    #Installs
    #spaCy: pip install spacy
    #English Model: python - m spacy download en_core_web_sm

    nlp = spacy.load("en_core_web_sm")
    #May not need: nlp.add_pipe("textblob")
    def prepareTweet(tweet): # Remove stopwords (and, a, an, etc), punctuation, special characters. convert everything to lower case.

        #Special characters, digits
        tweet = re.sub("(\\d|\\W)+"," ",tweet)

        #Make lowercase
        tweet = tweet.lower()

        #Tokenize tweet
        #Returns a doc object with processed info about tweet
        #https://spacy.io/usage/spacy-101: for stuff on doc object
        tweetDoc = nlp(tweet)

        #Remove Stopwords  and punctuation
        #https://spacy.io/api/token for .is_stop .is_punct
        processedTweet = [token.text for token in tweetDoc if not token.is_stop and not token.is_punct]

        finalProcessedTweet = " ".join(processedTweet)

        return finalProcessedTweet
    def calculateSentimentScore(tweet): #Calculate sentiment score
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe('spacytextblob')
        print("Calculating polarity for tweet: ", tweet)
        doc = nlp(tweet)
        #Return polarity score
        #Polarity Score: -1.0 (Negative)  to 1.0 (Positive)
        polarity = doc._.blob.polarity
        print("polarity: ", polarity)
        return polarity

    def categorizePolarity(score): #Categorizes the polarity as either Positive, Negative or neutral
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        else: return "neutral"


    sentiment_folder_name = 'Sentiment-By-Player'
    os.makedirs(sentiment_folder_name, exist_ok=True)

    mvp_tweets_path = os.path.join(os.path.dirname(__file__), 'Tag-Filtered-Tweets')

    # Go through player names in filtered tweets and analyze the sentiment for that player
    # Store count on the sentiments in 4 column csv (name, positive, neutral, negative)
    # Player object-oriented approach is an option too
    
    
    

    #Create a list to hold each voters dataframe.
    votersDataframesList = []

    #Filter through list of voters
    print("mvp_tag: ",  tags)
    for tag in tags:
        #Initialize Counts
        positive_tweets = 0
        neutral_tweets = 0
        negative_tweets = 0

        filename = f'{tag}_filtered_tweets.json'
        #print("Username: ", username)
        current_dir = os.getcwd()
        #print("filename ", filename)
        #print("Curr_dir", current_dir)



        filepath = os.path.join(mvp_tweets_path, filename)
        #print("filePath ", filepath)

        print("Reading tweets from {0}".format(tag))

        #Create voter's dataframe
        #Dataframe will hold: tweet, processedTweet, polarityScore, polarityCategory, and nominee/player
        df = pd.read_json(filepath)

        with open(filepath) as f:
            tweet_json = json.load(f)

        # Get filtered tweets from json file
        if not tweet_json:
            print("This file is empty")
        else:
            for item in tweet_json :
                if 'text' in item:
                    text_value = item['text']

                    #Filter tweet
                    print("original tweet: ", text_value)
                    processedTweet = prepareTweet(text_value)
                    print("processedTweet: ", processedTweet)
                    df["preparedTweet"] = processedTweet # df["preparedTweet"] = df["text"].apply(prepareTweet) #Create colomun for preparedTweets

                    #Calculate polarity score
                    polarityScore = calculateSentimentScore(processedTweet)
                    df["polarity"] = polarityScore #df["polarity"] = df["preparedTweet"].apply(calculateSentimentScore) #Create a colomun for Polarity scores

                    #Label positice, negative, or neutral
                    polarityCategory = categorizePolarity(polarityScore) #
                    df["sentiment"] = polarityCategory #df["sentiment"] = df["polarity"].apply(categorizePolarity) #Puts tweet in positive, negative, or neutral categories

                    #Find any nominee names in the tweets
                    people = [] # all people tagged in tweet
                    mentNominies = [] #mentioned athletes for a given tweet

                    #find all people tagged entities
                    doc = nlp(text_value)
                    for ent in doc.ents:
                        if (ent.label_ == "PERSON"):
                            people.append(ent.text)

                    #check if any noms are mentioned
                    for p in people:
                        if p in nomList:
                            mentNominies.append(p)
                    
                    #give out scores
                    for n in mentNominies:
                        mentNominies[n] += polarityScore

                    #Append dataFrame to voterslist
                    votersDataframesList.append(df)
                    print("voters dataframe list: \n", votersDataframesList)
                    print("dataframe: \n", df)

                    #Increment count
                    if polarityCategory == "positive":
                        print("positive")
                        positive_tweets += 1
                        print("increment positive, new positive_tweets: ", positive_tweets)
                    elif polarityCategory == "negative":
                        print("negative")
                        negative_tweets += 1
                        print("increment negative, new negative_tweets: ", negative_tweets)
                    elif polarityCategory == "neutral":
                        print("neutrals")
                        neutral_tweets += 1
                        print("increment neutral, new neutral_tweets: ", neutral_tweets)
        print("Positive Tweets: ", positive_tweets)
        print("Negative Tweets: ", negative_tweets)
        print("Neutral Tweets: ", neutral_tweets)
    # ------------------------------------------------------------------------------------------------------------------
    # Task 4
    # Calculate the sentiment score for each player by aggregating the sentiment of tweets mentioning them. Can use
    # different aggregation techniques such as mean or sum to calculate the sentiment score.
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Task 5
    # Identify the top players based on their sentiment scores and cross-check their performances with the statistics
    # such as points per game, rebounds per game, assists per game, etc.
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Task 6
    # Use machine learning algorithms to predict which player is most likely to be voted as the NBA regular season
    # MVP based on the broadcasters' votes and fan sentiment. Sentiment score and player statistics can be used as
    # input features for the machine learning model.
    # ------------------------------------------------------------------------------------------------------------------
