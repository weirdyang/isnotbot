import learningbot
import praw

from similarity import cosine_sim


def main():
    reddit = login_reddit()
    user = 'remindmebot'
    comments_list = reddit.redditor(user).comments.new(limit=5)
    input_list = [comment.body for comment in comments_list]
    #print(input_list)
    learningbot.testing_code(input_list)  
    learningbot.pipeline_cls(input_list)
    print(cosine_sim(input_list))

def login_reddit():
    reddit = praw.Reddit('isnotbot', user_agent='<Python> Test bot by u/captmomo')
    return reddit

if __name__ == '__main__':
    main()
