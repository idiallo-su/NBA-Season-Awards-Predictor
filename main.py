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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#from spacytextblob.spacytextblob import TextBlob
# from textblob import TextBlob

#import config

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
    start_date = datetime(2022, 10, 18)  # Start of the NBA regular season
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
            json.dump(filtered_tweets, f, indent="")
            print("{0} tweets dumped in {1}".format(username, filename))

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

    mvp_tweets_path = os.path.join(os.path.dirname(__file__), 'Voter-Filtered-Tweets')

    # Go through player names in filtered tweets and analyze the sentiment for that player
    # Store count on the sentiments in 4 column csv (name, positive, neutral, negative)
    # Player object-oriented approach is an option too

    #Create a list to hold each voters dataframe.
    votersDataframesList = []

    #Filter through list of voters
    print("mvp_voter_usernames: ",  mvp_voter_usernames)
    for username in mvp_voter_usernames:
        #Initialize Counts
        positive_tweets = 0
        neutral_tweets = 0
        negative_tweets = 0

        filename = f'{username[1:]}_filtered_tweets.json'
        #print("Username: ", username)
        current_dir = os.getcwd()
        #print("filename ", filename)
        #print("Curr_dir", current_dir)



        filepath = os.path.join(mvp_tweets_path, filename)
        #print("filePath ", filepath)

        print("Reading tweets from {0}".format(username))

        #Create voter's dataframe
        #Dataframe will hold: tweet, processedTweet, polarityScore, polarityCategory, and nominee/player
        df = pd.read_json(filepath)

        with open(filepath) as f:
            tweet_json = json.load(f)

        # Get filtered tweets from json file
        print("Current File: ", filename)
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

    # Prepare your data and # Split your data
    #X =  # Combine candidate stats, previous winners stats, sentiment of tweets, and team records into a single dataset
    #y =  # The target variable (whether or not the candidate won the MVP award)

    #2021-2022 season.
    #Player = [year, Teamseed, gamesPlayed, pts, rbs, assts, fg % ft %]

    nJokic2022 = [2022, 6, 74, 27.1, 13.8, 7.9, .583, .81]
    jEmbiid2022 = [2022, 4, 68, 30.6, 11.7, 4.2, .499, .814]
    gAntetokounmpo2022 = [2022, 3, 67, 29.9, 11.6, 5.8, .553, .722]
    dBooker2022 = [2022, 1, 68, 26.8, 5, 4.8, .466, .868]
    lDoncic2022 = [2022, 4, 65, 28.4, 9.1, 8.7, .457, .744]
    jTatum2022 = [2022, 2, 76, 26.9, 8, 4.4, .453, .853]
    jMorant2022 = [2022, 2, 57, 27.4, 5.7, 6.7, .493, .761]
    sCurry2022 = [2022, 3, 64, 25.5, 5.2, 6.3, .437, .923]
    cPaul2022 = [2022, 1, 65, 14.7, 4.4, 10.8, .493, .837]
    dDerozan2022 = [2022, 6, 76, 27.9, 5.2, 4.9, .504, .877]
    kDurant2022 = [2022, 7, 55, 29.9, 7.4, 6.4, .518, .91]
    lJames2022 = [2022, 11, 56, 30.3, 8.2, 6.2, .524, .756]

    #2019-2020
    gAntetokounmpo2020 = [2020, 1, 63, 29.5, 13.6, 5.6, .553, .633]
    lJames2020 = [2020, 1, 67, 25.3, 7.8, 10.2, .493, .693]
    jHarden2020 = [2020, 4, 68, 34.3, 6.6, 7.5, .444, .865]
    lDoncic2020 = [2020, 7, 61, 28.8, 9.4, 8.8, .463, .758]
    kLeonard2020 = [2020, 2, 57, 27.1, 7.1, 4.9, .47, .886]
    aDavis2020 = [2020, 1, 62, 26.1, 9.3, 3.2, .503, .846]
    cPaul2020 = [2020, 5, 70, 17.6, 5.0, 6.7, .489, .907]
    dLillard2020 = [2020, 8, 66, 30.0, 4.3, 8.0, .463, .888]
    nJokic2020 = [2020, 3, 73, 19.9, 9.7, 7.0, .528, .817]
    pSiakam2020 = [2020, 2, 60, 22.9, 7.3, 3.5, .453, .792]
    jButler2020 = [2020, 6, 58, 19.9, 6.7, 6.0, .455, .834]
    jTatum2020 = [2020, 3, 66, 23.4, 7.0, 3.0, .450, .812]

    #2018-2019
    gAntetokounmpo2019 = [2019, 1, 72, 27.7, 12.5, 5.9, .578, .729]
    jHarden2019 = [2019, 4, 78, 36.1, 6.6, 7.5, .442, .879]
    pGeorge2019 = [2019, 6, 77, 28, 8.2, 4.1, .438, .839]
    nJokic2019 = [2019, 2, 80, 20.1, 10.8, 7.3, .511, .821]
    sCurry2019 = [2019, 1, 69, 27.3, 5.3, 5.2, .472, .916]
    dLillard2019 = [2019, 3, 80, 25.8, 4.6, 6.9, .444, .912]
    jEmbiid2019 = [2019, 3, 64, 27.5, 13.6, 3.7, .484, .804]
    kDurant2019 = [2019, 1, 78, 26, 6.4, 5.9, .521, .885]
    kLeonard2019 = [2019, 2, 60, 26.6, 7.3, 3.3, .496, .854]
    rWestbrook2019 = [2019, 6, 73, 22.9, 11.1, 10.7, .428, .656]
    rGobert2019 = [2019, 5, 81, 15.9, 12.9, 2.0, .669, .636]
    lJames2019 = [2019, 10, 55, 27.4, 8.5, 8.3, .51, .665]

    #2013-2014
    kDurant2014 = [2014, 2, 81, 32, 7.4, 5.5, .503, .873]
    lJames2014 = [2014, 2, 77, 27.1, 6.9, 6.3, .567, .750]
    bGriffin2014 = [2014, 3, 80, 24.1, 9.5, 3.9, .528, .715]
    jNoah2014 = [2014, 4, 80, 12.6, 11.3, 5.4, .475, .737]
    jHarden2014 = [2014, 4, 73, 25.4, 4.7, 6.1, .456, .866]
    sCurry2014 = [2014, 6, 78, 24.0, 4.3, 8.5, .471, .885]
    cPaul2014 = [2014, 3, 62, 19.1, 4.3, 10.7, .467, .885]
    aJefferson2014 = [2014, 7, 73, 21.8, 10.8, 2.1, .509, .690]
    pGeorge2014 = [2014, 1, 80, 21.7, 6.8, 3.5, .424, .864]
    lAldridge2014 = [2014, 5, 69, 23.2, 11.1, 2.6, .458, .822]
    kLove2014 = [2014, 10, 77, 26.1, 12.5, 4.4, .457, .821]
    tDuncan2014 = [2014, 1, 74, 15.1, 9.7, 3.0, .490, .731]
    tParker2014 = [2014, 1, 68, 16.7, 2.3, 5.7, .499, .811]
    dNowitzki2014 = [2014, 8, 80, 21.7, 6.2, 2.7, .497, .899]
    cAnthony2014 = [2014, 9, 77, 27.4, 8.1, 3.1, .452, .848]
    gDragic2014 = [2014, 9, 76, 20.3, 3.2, 5.9, .505, .760]
    mConley2014 = [2014, 7, 73, 17.2, 2.9, 6.0, .450, .815]

    #Test Set
    x_train2022 = [nJokic2022, jEmbiid2022, gAntetokounmpo2022, dBooker2022, lDoncic2022, jTatum2022, jMorant2022, sCurry2022, cPaul2022, dDerozan2022, kDurant2022, lJames2022]
    y_train2022 = [True, False, False, False, False, False, False, False, False, False, False, False]
    x_train2020 = [gAntetokounmpo2020, lJames2020, jHarden2020, lDoncic2020, kLeonard2020, aDavis2020, cPaul2020, dLillard2020, nJokic2020, pSiakam2020, jButler2020, jTatum2020]
    y_train2020 = [True, False, False, False, False, False, False, False, False, False, False, False]
    x_train2019 = [gAntetokounmpo2019, jHarden2019, pGeorge2019, nJokic2019, sCurry2019, dLillard2019, jEmbiid2019, kDurant2019, kLeonard2019, rWestbrook2019, rGobert2019, lJames2019]
    y_train2019 = [True, False, False, False, False, False, False, False, False, False, False, False ]
    x_train2014 = [kDurant2014, lJames2014, bGriffin2014, jNoah2014, jHarden2014, sCurry2014, cPaul2014, aJefferson2014, pGeorge2014, lAldridge2014, kLove2014, tDuncan2014, tParker2014, dNowitzki2014, cAnthony2014, gDragic2014, mConley2014]
    y_train2014 = [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]

    x_train = x_train2020 + x_train2022
    y_train = y_train2020 + y_train2022

    #2020-2021 Season. Train set
    nJokic2021 = [2021, 3, 72, 26.4, 10.8, 8.3, .566, .868]
    jEmbiid2021 = [2021, 1, 51, 28.5, 10.6, 2.8, .513, .859]
    sCurry2021 = [2021, 8, 63, 32.0, 5.5, 5.8, .482, .916]
    gAntetokounmpo2021 = [2021, 3, 61, 28.1, 11, 5.9, .569, .685]
    cPaul2021 = [2021, 2, 70, 16.4, 4.5, 8.9, .499, .934]
    lDoncic2021 = [2021, 5, 66, 27.7, 8, 8.6, .479, .730]
    dLillard2021 = [2021, 6, 67, 28.8, 4.2, 7.5, .451, .928]
    jRandle2021 = [2021, 4, 71, 24.1, 10.2, 6, .456, .811]
    dRose2021 = [2021, 4, 59, 14.7, 2.6, 4.2, .47, .866]
    rGobert2021 = [2021, 1, 71, 14.3, 13.5, 1.3, .675, .623]
    rWestbrook2021 = [2021, 8, 65, 22.2, 11.5, 11.7, .439, .656]
    bSimmons2021 = [2021, 1, 58, 14.3, 7.2, 6.9, .557, .613]
    jHarden2021 = [2021, 2, 44, 24.6, 7.9, 10.8, .466, .861]
    lJames2021 = [2021, 7, 45, 25.0, 7.7, 7.8, .513, .698]
    kLeonard2021 = [2021, 4, 52, 24.8, 6.5, 5.2, .512, .885]



    #Train Set
    x_test = [nJokic2021, jEmbiid2021, sCurry2021, gAntetokounmpo2021, cPaul2021, lDoncic2021, dLillard2021, jRandle2021, dRose2021, rGobert2021, rWestbrook2021, bSimmons2021, jHarden2021, lJames2021, kLeonard2021]
    y_test = [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]



    # Train model
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    # Evaluate  model
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n\n\nTest set Accuracy: {accuracy:.2f}")

    #2023 Candidates
    # Player = [Teamseed, gamesPlayed, pts, rbs, assts, fg % ft %]
    jEmbiid2023 = [2023, 3, 66, 33.1, 10.2, 4.2, .548, .857]
    gAntetokounmpo2023 = [2023, 1, 63, 31.1, 11.8, 5.7, .553, .645]
    nJokic2023 = [2023, 1, 69, 24.5, 11.8, 9.8, .632, .822]

    #Predict the winner
    players = [jEmbiid2023, gAntetokounmpo2023, nJokic2023]
    predictions = []
    for player in players:
        prediction = clf.predict([player])
        predictions.append(bool(prediction[0]))

    # Print the list of predictions
    print("This years Predictions: ", predictions)