# main.py
from datetime import datetime
from http.client import BadStatusLine
from urllib.error import URLError

import twitter
import json
import csv
import os
import sys
import time
import spacy
from spacytextblob.spacytextblob import TextBlob  # https://pypi.org/project/spacytextblob/

import config

from TwitterCookbook import oauth_login, make_twitter_request, harvest_user_timeline


if __name__ == '__main__':

    twitter_api = oauth_login()

    print("Twitter Api {0}\n".format(twitter_api))

    # ------------------------------------------------------------------------------------------------------------------
    # Task 2
    # Use Twitter's API to collect tweets from fans about NBA players' performances throughout the regular
    # season. Use relevant hashtags or keywords to filter tweets specifically about NBA players.
    # ------------------------------------------------------------------------------------------------------------------

    mvp_voter_path = os.path.join(os.path.dirname(__file__), 'NBA-Season-Award-Voters/MVP Voters By Year.csv')

    # Get voter usernames from csv file
    with open(mvp_voter_path, newline='') as mvp_voter_file:
        reader = csv.reader(mvp_voter_file)
        mvp_voter_usernames = []

        for i, row in enumerate(reader):
            if row[7] != '':
                mvp_voter_usernames.append(row[7])

    print(mvp_voter_usernames, '\n')

    # Date range to harvest tweets from
    # start_date = datetime(2022, 10, 18)  # Start of the NBA regular season
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 4, 9)  # Can push to a later date

    tweets_folder_name = 'Voter-Filtered-Tweets'
    os.makedirs(tweets_folder_name, exist_ok=True)

    # Harvest tweets from each voter
    for username in mvp_voter_usernames:
        print(username)
        username = username[1:]  # Remove '@' sign

        tweets = harvest_user_timeline(twitter_api, screen_name=username, max_results=2)

        print("{0} tweets: ".format(username))
        print(tweets)

        # Filter tweets by date and possibly player names
        filtered_tweets = []
        for tweet in tweets:
            created_at_str = tweet['created_at']
            created_at = datetime.strptime(created_at_str, '%a %b %d %H:%M:%S +0000 %Y')
            # print("Tweet created at {0}\n".format(created_at))

            if start_date <= created_at <= end_date:
                filtered_tweets.append(tweet)

        print("{0} filtered tweets: ".format(username))
        print(filtered_tweets)

        # Save the filtered tweets to a json file within Voter-Filtered-Tweets
        filename = f'{username}_filtered_tweets.json'
        filepath = os.path.join(tweets_folder_name, filename)
        with open(filepath, 'w') as f:
            json.dump(filtered_tweets, f, indent=4)
            print("{0} tweets dumped in {1}".format(username, filename))

    # ------------------------------------------------------------------------------------------------------------------
    # Task 3
    # Analyze the sentiment of the tweets using natural language processing (NLP) techniques. NLP libraries: NLTK or
    # spaCy to identify the sentiment of the tweets (positive, negative, or neutral) towards specific players.
    # ------------------------------------------------------------------------------------------------------------------

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textblob")

    sentiment_folder_name = 'Sentiment-By-Player'
    os.makedirs(sentiment_folder_name, exist_ok=True)

    mvp_tweets_path = os.path.join(os.path.dirname(__file__), 'Voter-Filtered-Tweets')

    # Go through player names in filtered tweets and analyze the sentiment for that player
    # Store count on the sentiments in 4 column csv (name, positive, neutral, negative)
    # Player object-oriented approach is an option too
    for username in mvp_voter_usernames:
        filename = f'{username}_filtered_tweets.json'
        filepath = os.path.join(mvp_tweets_path, filename)

        positive_tweets = 0
        neutral_tweets = 0
        negative_tweets = 0

        # Get filtered tweets from json file
        # for ___ in ___ :
        #     doc = nlp(tweet)
        #     sentiment = doc._.polarity

            # Increment variable based on sentiment

        # Save sentiment data for this player

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
