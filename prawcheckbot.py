import learningbot

from prawbot import login_reddit, answers_scrape


def main():
    reddit = login_reddit()
    url = 'https://www.reddit.com/r/AskReddit/comments/7zzim3/what_app_is_so_useful_you_cant_believe_its_free/'
    user = 'automoderator'
    comments_list = reddit.redditor(user).comments.new(limit=500)
    input_list = [comment.body for comment in comments_list]
    learningbot.pipeline_cls(input_list)

if __name__ == '__main__':
    main()
