import ijson
import re
import html
import string
import pickle
import numpy as np

import nltk
from nltk.corpus import stopwords
# nltk.download('wordnet')
# nltk.download('stopwords')

def get_data(path, url_regex):
    lemmatizer = nltk.wordnet.WordNetLemmatizer()
    subject_database = {}
    with open(path, 'rb') as input_file:
        for item in ijson.items(input_file, 'item'):
            id, label = item['ID'], item['label']
            if item['tweet'] is not None:
                tweets = []
                for tweet in item['tweet']:
                    s = tweet.strip().encode('ascii', 'ignore').decode('ascii') # eliminate emojis
                    s = re.sub(r'\s*(RT)\s+@[A-Z0-9a-z_:]+\s*','',s) # replace RT @User
                    s = re.sub(r'@[A-Z0-9a-z_:]+','',s) # replace user tags       
                    s = re.sub(url_regex,'',s) # replace URLs, may not be perfect
                    s = re.sub(r'\s*#[a-zA-Z]+\s*','',s) # replace hashtags
                    s = html.unescape(s) # decode HTML character codes
                    s = s.translate(str.maketrans('', '', string.punctuation)) # puctuation
                    s = re.sub(r' {2,}',' ',s).lower() # eliminate extra spaces
                    s = ' '.join([lemmatizer.lemmatize(x) for x in s.split() if x not in stopwords.words('english')]) # Lemmatization & stop words

                    if s != '':
                        # print(s)
                        tweets.append(s)

                subject_database[id] = [label, tweets]
    return subject_database

def split_train_val(path, url_regex):
    train_db = get_data(path, url_regex)
    ratio = 0.1

    # Split Bots
    bot_ids = [k for k,v in train_db.items() if v[0] == '1']
    bot_val_ids = [bot_ids[i] for i in np.random.choice(len(bot_ids), size=int(len(bot_ids) * ratio), replace=False)]
    bot_train_ids = [v for v in bot_ids if v not in bot_val_ids]
    bot_train_data, bot_val_data = {k:v for k,v in train_db.items() if k in bot_train_ids}, {k:v for k,v in train_db.items() if k in bot_val_ids}

    # Split Reals
    real_ids = [k for k in train_db if k not in bot_ids]
    real_val_ids = [real_ids[i] for i in np.random.choice(len(real_ids), size=int(len(real_ids) * ratio), replace=False)]
    real_train_ids = [v for v in real_ids if v not in real_val_ids]
    real_train_data, real_val_data = {k:v for k,v in train_db.items() if k in real_train_ids}, {k:v for k,v in train_db.items() if k in real_val_ids}

    # Merge
    train_data, val_data = bot_train_data | real_train_data, bot_val_data | real_val_data
    
    with open('../database/train_filtered.pkl', 'wb') as file:
        pickle.dump(train_data, file)

    with open('../database/val_filtered.pkl', 'wb') as file:
        pickle.dump(val_data, file)

if __name__ == '__main__':
    # emoji.replace_emoji("Hi, I am fine. 😁💀", replace='')
    # url_regex = r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"
    # url_regex = "((?<=[^a-zA-Z0-9])(?:https?\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)(?:\w{1,}\.{1}){1,5}(?:com|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|mil|iq|io|ac|ly|sm){1}(?:\/[a-zA-Z0-9]{1,})*)"
    url_regex = r'(https\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)(?:\w{1,}\.{1}){1,5}(?:com|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|mil|iq|io|ac|ly|co|sm){1}(?:\/[a-zA-Z0-9]{1,})*'

    train_path = '../database/train.json'
    test_path = '../database/test.json'
    
    # Do train & val
    split_train_val(train_path, url_regex)

    # Do test, simpler
    test_data = get_data(test_path, url_regex)
    with open('../database/test_filtered.pkl', 'wb') as file:
        pickle.dump(test_data, file)

    _ = 1



    
    

    

    
    