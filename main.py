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
    tags = ["jokic", "giannis", "embiid", "mvpnba", "nba"]

    # Harvest tweets from each voter
    for t in tags:
        tweets = twitter_search(twitter_api, t, max_results=20)

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

    #Find count for each players individual count increment


    #Filter through list of voters
    print("mvp_tag: ",  tags)
    # Initialize Counts
    totalPositive = 0
    totalNegative = 0
    totalNeutral = 0

    embiidPositive = 0
    embiidNegative = 0
    embiidNeutral = 0

    giannisPositive = 0
    giannisNegative = 0
    giannisNeutral = 0

    jokicPositive = 0
    jokicNegative = 0
    jokicNeutral = 0

    count = 0

    for tag in tags:


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
                    count += 1
                    print('count: ', count)

                    #Filter tweet                           #print("original tweet: ", text_value)
                    processedTweet = prepareTweet(text_value)

                                                            #print("processedTweet: ", processedTweet)


                    #Calculate polarity score
                    polarityScore = calculateSentimentScore(processedTweet)


                    #Label positice, negative, or neutral
                    polarityCategory = categorizePolarity(polarityScore)

                    # df["preparedTweet"] = processedTweet # df["preparedTweet"] = df["text"].apply(prepareTweet) #Create colomun for preparedTweets
                    # df["polarity"] = polarityScore #df["polarity"] = df["preparedTweet"].apply(calculateSentimentScore) #Create a colomun for Polarity scores
                    # df["sentiment"] = polarityCategory #df["sentiment"] = df["polarity"].apply(categorizePolarity) #Puts tweet in positive, negative, or neutral categories

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
                        print("[Mentioned]: ", n)

                        #update Polarity
                        nomList[n] += polarityScore
                        print(n,"'s new score: ", nomList[n] )

                        #Increment counts
                        if polarityCategory == "positive":
                            print("positive")
                            totalPositive += 1
                            if n == "Giannis":
                                giannisPositive += 1
                                print("increment giannisPositive, new positive_tweets: ", giannisPositive)
                            if n == "Joel Embiid":
                                embiidPositive += 1
                                print("increment embiidPositive, new positive_tweets: ", embiidPositive)
                            if n == "Nikola Jokic":
                                jokicPositive += 1
                                print("increment jokicPositive, new positive_tweets: ", jokicPositive)

                        elif polarityCategory == "negative":
                            totalNegative += 1
                            if n == "Giannis":
                                giannisNegative += 1
                                print("increment giannisNegative, new positive_tweets: ", giannisNegative)
                            if n == "Joel Embiid":
                                embiidNegative += 1
                                print("increment embiidNegative, new positive_tweets: ", embiidNegative)
                            if n == "Nikola Jokic":
                                jokicNegative += 1
                                print("increment jokicNegative, new positive_tweets: ", jokicNegative)

                        elif polarityCategory == "neutral":
                            totalNeutral += 1
                            if n == "Giannis":
                                giannisNeutral += 1
                                print("increment giannisNeutral, new positive_tweets: ", giannisNeutral)
                            if n == "Joel Embiid":
                                embiidNeutral += 1
                                print("increment embiidNeutral, new positive_tweets: ", embiidNeutral)
                            if n == "Nikola Jokic":
                                jokicNegative += 1
                                print("increment jokicNegative, new positive_tweets: ", jokicNegative)

                    #Append dataFrame to voterslist
                    votersDataframesList.append(df)
                    #print("voters dataframe list: \n", votersDataframesList)
                    #print("dataframe: \n", df)

    print("************Final Print************")
    print("\nGiannis\n")
    print("Giannis Positive Tweets: ", giannisPositive)
    print("Giannis Negative Tweets: ", giannisNegative)
    print("Giannis Neutral Tweets: ", giannisNeutral)

    print("\nEmbiid\n")
    print("Embiid Positive Tweets: ", embiidPositive)
    print("Embiid Negative Tweets: ", embiidNegative)
    print("Embiid Neutral Tweets: ", embiidNeutral)

    print("\nJokic\n")
    print("Jokic Positive Tweets: ", jokicPositive)
    print("Jokic Negative Tweets: ", jokicNegative)
    print("Jokic Neutral Tweets: ", jokicNeutral)

    print("\nTotal Positive: ", totalPositive)
    print("Total Negative: ", totalNegative)
    print("Total Neutral: ", totalNeutral)
    totalTweets = totalPositive + totalNegative + totalNeutral
    print("Total Tweets Collected: ", totalTweets)
    print("TOTAL TWEET COUNT: ", count)

    print("\nnomList: ", nomList, '\n')





    # ------------------------------------------------------------------------------------------------------------------
    # Task 4
    # Calculate the sentiment score for each player by aggregating the sentiment of tweets mentioning them. Can use
    # different aggregation techniques such as mean or sum to calculate the sentiment score.
    # ------------------------------------------------------------------------------------------------------------------
    ranking = [nomList["Joel Embiid"] , nomList["Giannis"], nomList["Nikola Jokic"]]
    players = ["Joel Embiid", "Giannis", "Nikola Jokic"]
    first = None
    firstScore = -10000
    second = None
    secondScore = -10000
    third = None
    thirdScore = -10000
    for i in range(3):
        #If new top sentiment is found
        if ranking[i] > firstScore:
            #Move intial first down to second
            second = first
            secondScore = firstScore
            #Move initial second down to third
            third = second
            thirdScore = secondScore
            #Replace new first
            first = players[i]
            firstScore = ranking[i]
        #If player isnt top sentiment, but may be second
        elif ranking[i] > secondScore:
            #Move initial second down to third
            third = second
            thirdScore = secondScore
            #Replace second
            second = players[i]
            secondScore = ranking[i]
        else:
            third = players[i]
            thirdScore = ranking[i]

    print("\nFirst: ", first, " Second: ", second, " Third: ", third)

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


    #2023 Candidates
    # Player = [Teamseed, gamesPlayed, pts, rbs, assts, fg % ft %]
    jEmbiid2023 = [3, 66, 33.1, 10.2, 4.2, .548, .857]
    gAntetokounmpo2023 = [1, 63, 31.1, 11.8, 5.7, .553, .645]
    nJokic2023 = [1, 69, 24.5, 11.8, 9.8, .632, .822]

    #Apply points multiplier
    if first == "Joel Embiid":
        jEmbiid2023[2] *= 1.1
    elif first == "Giannis":
        gAntetokounmpo2023[2] *= 1.1
    else:
        nJokic2023[2] *= 1.1

    if second == "Joel Embiid":
        jEmbiid2023[2] *= 1.05
    elif second == "Giannis":
        gAntetokounmpo2023[2] *= 1.05
    else:
        nJokic2023[2] *= 1.05

    #Training Set
    #2021-2022 season.
    #Player = [year, Teamseed, gamesPlayed, pts, rbs, assts, fg % ft %]
    nJokic2022 = [6, 74, 27.1, 13.8, 7.9, .583, .81]
    jEmbiid2022 = [4, 68, 30.6, 11.7, 4.2, .499, .814]
    gAntetokounmpo2022 = [3, 67, 29.9, 11.6, 5.8, .553, .722]
    dBooker2022 = [1, 68, 26.8, 5, 4.8, .466, .868]
    lDoncic2022 = [4, 65, 28.4, 9.1, 8.7, .457, .744]
    jTatum2022 = [2, 76, 26.9, 8, 4.4, .453, .853]
    jMorant2022 = [2, 57, 27.4, 5.7, 6.7, .493, .761]
    sCurry2022 = [3, 64, 25.5, 5.2, 6.3, .437, .923]
    cPaul2022 = [1, 65, 14.7, 4.4, 10.8, .493, .837]
    dDerozan2022 = [6, 76, 27.9, 5.2, 4.9, .504, .877]
    kDurant2022 = [7, 55, 29.9, 7.4, 6.4, .518, .91]
    lJames2022 = [11, 56, 30.3, 8.2, 6.2, .524, .756]

    x_train = [nJokic2022, jEmbiid2022, gAntetokounmpo2022, dBooker2022, lDoncic2022, jTatum2022, jMorant2022, sCurry2022, cPaul2022, dDerozan2022, kDurant2022, lJames2022]
    y_train = [True, False, False, False, False, False, False, False, False, False, False, False]

    #Testing Set
    #2020-2021 Season. Train set
    nJokic2021 = [3, 72, 26.4, 10.8, 8.3, .566, .868]
    jEmbiid2021 = [1, 51, 28.5, 10.6, 2.8, .513, .859]
    sCurry2021 = [8, 63, 32.0, 5.5, 5.8, .482, .916]
    gAntetokounmpo2021 = [ 3, 61, 28.1, 11, 5.9, .569, .685]
    cPaul2021 = [ 2, 70, 16.4, 4.5, 8.9, .499, .934]
    lDoncic2021 = [ 5, 66, 27.7, 8, 8.6, .479, .730]
    dLillard2021 = [6, 67, 28.8, 4.2, 7.5, .451, .928]
    jRandle2021 = [ 4, 71, 24.1, 10.2, 6, .456, .811]
    dRose2021 = [4, 59, 14.7, 2.6, 4.2, .47, .866]
    rGobert2021 = [1, 71, 14.3, 13.5, 1.3, .675, .623]
    rWestbrook2021 = [8, 65, 22.2, 11.5, 11.7, .439, .656]
    bSimmons2021 = [1, 58, 14.3, 7.2, 6.9, .557, .613]
    jHarden2021 = [2, 44, 24.6, 7.9, 10.8, .466, .861]
    lJames2021 = [7, 45, 25.0, 7.7, 7.8, .513, .698]
    kLeonard2021 = [4, 52, 24.8, 6.5, 5.2, .512, .885]

    x_test = [nJokic2021, jEmbiid2021, sCurry2021, gAntetokounmpo2021, cPaul2021, lDoncic2021, dLillard2021, jRandle2021, dRose2021, rGobert2021, rWestbrook2021, bSimmons2021, jHarden2021, lJames2021, kLeonard2021]
    y_test = [True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]

    # Train model
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    # Evaluate  model
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n\nTest set Accuracy: {accuracy:.2f}")

    # Predict the winner
    players = [jEmbiid2023, gAntetokounmpo2023, nJokic2023]
    predictions = []
    for player in players:
        prediction = clf.predict([player])
        predictions.append(bool(prediction[0]))

    # Print the list of predictions

    print("This years Predictions: \n", ["Embiid", "Giannis", "Jokic"], '\n', predictions)



