

import twitter
import json

from TwitterCookbook import oauth_login, harvest_user_timeline

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    twitter_api = oauth_login()

    print("Twitter Api {0}\n".format(twitter_api))

    # Task 2
    # Use Twitter's API to collect tweets from fans about NBA players' performances throughout the regular
    # season. Use relevant hashtags or keywords to filter tweets specifically about NBA players.

    # Task 3
    # Analyze the sentiment of the tweets using natural language processing (NLP) techniques. NLP libraries: NLTK or
    # spaCy to identify the sentiment of the tweets (positive, negative, or neutral) towards specific players.

    # Task 4
    # Calculate the sentiment score for each player by aggregating the sentiment of tweets mentioning them. Can use
    # different aggregation techniques such as mean or sum to calculate the sentiment score.

    # Task 5
    # Identify the top players based on their sentiment scores and cross-check their performances with the statistics
    # such as points per game, rebounds per game, assists per game, etc.

    # Task 6
    # Use machine learning algorithms to predict which player is most likely to be voted as the NBA regular season
    # MVP based on the broadcasters' votes and fan sentiment. Sentiment score and player statistics can be used as
    # input features for the machine learning model.
