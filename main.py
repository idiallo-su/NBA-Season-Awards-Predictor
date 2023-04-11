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

import config

# from TwitterCookbook import oauth_login, make_twitter_request, harvest_user_timeline


# Twitter Cookbook: Chapter 9 - Example 1. Accessing Twitter's API for development purposes
def oauth_login():
    # XXX: Go to http://twitter.com/apps/new to create an app and get values
    # for these credentials that you'll need to provide in place of these
    # empty string values that are defined as placeholders.
    # See https://developer.twitter.com/en/docs/basics/authentication/overview/oauth
    # for more information on Twitter's OAuth implementation.

    CONSUMER_KEY = config.CONSUMER_KEY
    CONSUMER_SECRET = config.CONSUMER_SECRET
    OAUTH_TOKEN = config.OAUTH_TOKEN
    OAUTH_TOKEN_SECRET = config.OAUTH_TOKEN_SECRET

    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)

    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


# Twitter Cookbook: Chapter 9 - Example 16. Making robust Twitter requests
def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw):
    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting issue (429 error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):

        if wait_period > 3600:  # Seconds
            print('Too many retries. Quitting.', file=sys.stderr)
            raise e

        # See https://developer.twitter.com/en/docs/basics/response-codes
        # for common codes

        if e.e.code == 401:
            print('Encountered 401 Error (Not Authorized)', file=sys.stderr)
            return None
        elif e.e.code == 404:
            print('Encountered 404 Error (Not Found)', file=sys.stderr)
            return None
        elif e.e.code == 429:
            print('Encountered 429 Error (Rate Limit Exceeded)', file=sys.stderr)
            if sleep_when_rate_limited:
                print("Retrying in 15 minutes...ZzZ...", file=sys.stderr)
                sys.stderr.flush()
                time.sleep(60 * 15 + 5)
                print('...ZzZ...Awake now and trying again.', file=sys.stderr)
                return 2
            else:
                raise e  # Caller must handle the rate limiting issue
        elif e.e.code in (500, 502, 503, 504):
            print('Encountered {0} Error. Retrying in {1} seconds'.format(e.e.code, wait_period), file=sys.stderr)
            time.sleep(wait_period)
            wait_period *= 1.5
            return wait_period
        else:
            raise e

    # End of nested helper function

    wait_period = 2
    error_count = 0

    while True:
        try:
            return twitter_api_func(*args, **kw)
        except twitter.api.TwitterHTTPError as e:
            error_count = 0
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("URLError encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise
        except BadStatusLine as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print("BadStatusLine encountered. Continuing.", file=sys.stderr)
            if error_count > max_errors:
                print("Too many consecutive errors...bailing out.", file=sys.stderr)
                raise


# Twitter Cookbook: Chapter 9 - Example 21. Harvesting a user's tweets
# *** Tweet count adjusted in here ***
def harvest_user_timeline(twitter_api, screen_name=None, user_id=None, max_results=1000):
    assert (screen_name != None) != (user_id != None), "Must have screen_name or user_id, but not both"

    kw = {  # Keyword args for the Twitter API call
        'count': 20,
        'trim_user': 'true',
        'include_rts': 'true',
        'since_id': 1
    }

    if screen_name:
        kw['screen_name'] = screen_name
    else:
        kw['user_id'] = user_id

    max_pages = 16
    results = []

    tweets = make_twitter_request(twitter_api.statuses.user_timeline, **kw)

    if tweets is None:  # 401 (Not Authorized) - Need to bail out on loop entry
        tweets = []

    results += tweets

    print('Fetched {0} tweets'.format(len(tweets)), file=sys.stderr)

    page_num = 1

    # Many Twitter accounts have fewer than 200 tweets so you don't want to enter
    # the loop and waste a precious request if max_results = 200.

    # Note: Analogous optimizations could be applied inside the loop to try and
    # save requests. e.g. Don't make a third request if you have 287 tweets out of
    # a possible 400 tweets after your second request. Twitter does do some
    # post-filtering on censored and deleted tweets out of batches of 'count', though,
    # so you can't strictly check for the number of results being 200. You might get
    # back 198, for example, and still have many more tweets to go. If you have the
    # total number of tweets for an account (by GET /users/lookup/), then you could
    # simply use this value as a guide.

    if max_results == kw['count']:
        page_num = max_pages  # Prevent loop entry

    while page_num < max_pages and len(tweets) > 0 and len(results) < max_results:
        # Necessary for traversing the timeline in Twitter's v1.1 API:
        # get the next query's max-id parameter to pass in.
        # See https://dev.twitter.com/docs/working-with-timelines.
        kw['max_id'] = min([tweet['id'] for tweet in tweets]) - 1

        tweets = make_twitter_request(twitter_api.statuses.user_timeline, **kw)
        results += tweets

        print('Fetched {0} tweets'.format(len(tweets)), file=sys.stderr)

        page_num += 1

    print('Done fetching tweets', file=sys.stderr)

    return results[:max_results]


# Press the green button in the gutter to run the script.
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

    folder_name = 'Voter-Filtered-Tweets'
    os.makedirs(folder_name, exist_ok=True)

    # Harvest tweets from each voter
    for username in mvp_voter_usernames:
        print(username)
        username = username[1:] # Remove '@' sign

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
        filepath = os.path.join(folder_name, filename)
        with open(filepath, 'w') as f:
            json.dump(filtered_tweets, f, indent=4)
            print("{0} tweets dumped in {1}".format(username, filename))

    # ------------------------------------------------------------------------------------------------------------------
    # Task 3
    # Analyze the sentiment of the tweets using natural language processing (NLP) techniques. NLP libraries: NLTK or
    # spaCy to identify the sentiment of the tweets (positive, negative, or neutral) towards specific players.
    # ------------------------------------------------------------------------------------------------------------------

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
