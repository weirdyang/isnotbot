import learningbot

from prawbot import login_reddit, answers_scrape
from similarity import cosine_sim


def main():
    reddit = login_reddit()
    url = 'https://www.reddit.com/r/AskReddit/comments/7zzim3/what_app_is_so_useful_you_cant_believe_its_free/'
    user = 'skoetje'
    comments_list = reddit.redditor(user).comments.new(limit=10)
    input_list = [comment.body for comment in comments_list]
    #print(input_list)
    learningbot.pipeline_cls(input_list)
    learningbot.testing_code(input_list)
    print(cosine_sim(input_list))

if __name__ == '__main__':
    main()
