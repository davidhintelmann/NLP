# Natural Language Processing

This notebook starts by using [PRAW](https://praw.readthedocs.io/en/latest/#), \"The Python Reddit API Wrapper\" which will be used to download comments from the subreddit [r/news/](https://www.reddit.com/r/news/).
    
PRAW can be used to create chat bots on reddit or just to scrap data from it to gain insights into online social media. I will use to [Scapy](https://scapy.readthedocs.io/en/latest/introduction.html) to then perform natural language processing since this python library already have pretain models for this task.

We will try to create an algorithm to detect online harassment, and in particular to flag if a comment has a high likihood of contain hate speech.

## Issue Reddit Api Token

Go to this [page](https://www.reddit.com/prefs/apps) to create an app on Reddit's API page.
Rules for Reddits API can be found [here](https://github.com/reddit-archive/reddit/wiki/API).
Instructions for creating Reddit app below have been taken from [Felippe Rodrigues's](https://www.storybench.org/how-to-scrape-reddit-with-python/) post from [storybench.org](https://www.storybench.org/)
![image](https://www.storybench.org/wp-content/uploads/2018/03/Screen-Shot-2018-02-28-at-5.37.01-PM.png)

This form should open up:

![image](https://www.storybench.org/wp-content/uploads/2018/03/Screen-Shot-2018-02-28-at-6.55.38-PM.png)
Pick a name for your application and add a description for reference. Also make sure you select the “script” option and don’t forget to put `http://localhost:8080` in the redirect uri field. If you have any doubts, refer to [Praw documentation](https://praw.readthedocs.io/en/latest/getting_started/authentication.html#script-application). 

Hit create app and now you are ready to use the OAuth2 authorization to connect to the API and start scraping. Copy and paste your 14-characters personal use script and 27-character secret key somewhere safe. You application should look like this:
![image](https://www.storybench.org/wp-content/uploads/2018/03/Screen-Shot-2018-02-28-at-7.02.45-PM.png)