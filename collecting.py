# -*- coding: utf-8 -*-
import tweepy
from datetime import date

consumer_key = "uwQ0cv9YIySsANx67TjntFybt"
consumer_secret = "Cqv67ISNsSRXXncTlLcE1BRWc6DCu1smollvVLFR94NyGlqi2N"
access_token = "1037713205248704513-5XpRSL5ufznW3qMkCMakdgYPHd7HlB"
access_token_secret = "1g1tJkH7jIWUi144fjF7hFsqL3aUFSJJDk9Vkofw0itgz"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

today = date.today() 
name = 'data-' + today.strftime("%m%d") + '.txt'
f = open(name, 'w', encoding="utf-8")

tweets = tweepy.Cursor(api.search,
                        q="#เลือกตั้ง62",
                        lang="th",
                        since="2019-01-01",
                        result_type="mixed",
                        include_entities="false",
                        tweet_mode="extended").items()
for tweet in tweets:
    if 'retweeted_status' in dir(tweet):
        record = tweet.retweeted_status.full_text
        f.write(record)
    else:
        record = tweet.full_text
        f.write(record)
f.close()