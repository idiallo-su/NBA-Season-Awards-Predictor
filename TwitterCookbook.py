#!/usr/bin/env python
# coding: utf-8

# # Mining the Social Web, 3rd Edition
#
# ## Chapter 9: Twitter Cookbook
#
# This Jupyter Notebook provides an interactive way to follow along with and explore the numbered examples from [_Mining the Social Web (2nd Edition)_]
# (http://bit.ly/135dHfs). The intent behind this notebook is to reinforce the concepts from the sample code in a fun, convenient, and effective way. This
# notebook assumes that you are reading along with the book and have the context of the discussion as you work through these exercises.
#
# In the somewhat unlikely event that you've somehow stumbled across this notebook outside of its context on GitHub, [you can find the full source code
# repository here](http://bit.ly/MiningTheSocialWeb3E).
#
# ## Copyright and Licensing
#
# You are free to use or adapt this notebook for any purpose you'd like. However, please respect the [Simplified BSD License]
# (https://github.com/ptwobrussell/Mining-the-Social-Web-2nd-Edition/blob/master/LICENSE.txt) that governs its use.
#
# ## Notes
#
# This notebook is still a work in progress and currently features 25 recipes. The example titles should be fairly self-explanatory, and the code is designed
# to be reused as you progress further in the notebook --- meaning that you should follow along and execute each cell along the way since later cells may depend
# on functions being defined from earlier cells. Consider this notebook draft material at this point.

from datetime import datetime
from http.client import BadStatusLine
from urllib.error import URLError
from urllib.parse import unquote

import twitter
import json
import csv
import os
import sys
import time

#import config


# Twitter Cookbook: Chapter 9 - Example 1. Accessing Twitter's API for development purposes
def oauth_login():
    # XXX: Go to http://twitter.com/apps/new to create an app and get values
    # for these credentials that you'll need to provide in place of these
    # empty string values that are defined as placeholders.
    # See https://developer.twitter.com/en/docs/basics/authentication/overview/oauth
    # for more information on Twitter's OAuth implementation.

    CONSUMER_KEY = 'AIZqbzPyEPxtokxdKFnozPCNe'
    CONSUMER_SECRET = 'sZxFPtdwQPPjG7ytGsCJDQrAoh6LXbFQ1K0A3ZDar0jWvywY5l'
    OAUTH_TOKEN = '1496910867119624194-2QGEZqA55ufL4IriJKGTvL9p39P5g2'
    OAUTH_TOKEN_SECRET = 'GiNwfVYgfjnWONo5NhVf2G3wmklirPNibUrFf8Nsku0rn'

    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)

    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api

def twitter_search(twitter_api, q, max_results=200, **kw):

    # See https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets
    # and https://developer.twitter.com/en/docs/tweets/search/guides/standard-operators
    # for details on advanced search criteria that may be useful for 
    # keyword arguments
    
    # See https://dev.twitter.com/docs/api/1.1/get/search/tweets    
    search_results = twitter_api.search.tweets(q=q, count=100, **kw)
    
    statuses = search_results['statuses']
    
    # Iterate through batches of results by following the cursor until we
    # reach the desired number of results, keeping in mind that OAuth users
    # can "only" make 180 search queries per 15-minute interval. See
    # https://developer.twitter.com/en/docs/basics/rate-limits
    # for details. A reasonable number of results is ~1000, although
    # that number of results may not exist for all queries.
    
    # Enforce a reasonable limit
    max_results = min(1000, max_results)
    
    for _ in range(10): # 10*100 = 1000
        try:
            next_results = search_results['search_metadata']['next_results']
        except KeyError as e: # No more results when next_results doesn't exist
            break
            
        # Create a dictionary from next_results, which has the following form:
        # ?max_id=313519052523986943&q=NCAA&include_entities=1
        kwargs = dict([ kv.split('=') for kv in unquote(next_results[1:]).split("&") ])
        
        search_results = twitter_api.search.tweets(**kwargs)
        statuses += search_results['statuses']
        
        if len(statuses) >= max_results: 
            break
            
    return statuses

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



