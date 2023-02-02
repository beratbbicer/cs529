import pickle
import numpy as np

# nltk.download('wordnet')
# nltk.download('stopwords')

def get_word_dicts(dict):
    bot_dict, real_dict = {},{}
    for _, v in dict.items():
        label, tweets = v[0], v[1]
        for tweet in tweets:
            for word in tweet.split():
                if label == '0':
                    if word in real_dict:
                        real_dict[word] += 1
                    else:
                        real_dict[word] = 1
                else:
                    if word in bot_dict:
                        bot_dict[word] += 1
                    else:
                        bot_dict[word] = 1
    return bot_dict, real_dict

if __name__ == '__main__':
    train_path = '../database/train_filtered.pkl'

    with open(train_path, 'rb') as file:
        subject_dict_train = pickle.load(file) # 7401

    # Do train & val
    bot_dict, real_dict = get_word_dicts(subject_dict_train)
    bot_dict = {k: v for k, v in sorted(bot_dict.items(), key=lambda item: item[1], reverse=True)}
    real_dict = {k: v for k, v in sorted(real_dict.items(), key=lambda item: item[1], reverse=True)}
    
    with open('../database/bot_dictionary_train.pkl', 'wb') as file:
        pickle.dump(bot_dict, file)

    with open('../database/real_dictionary_train.pkl', 'wb') as file:
        pickle.dump(real_dict, file)