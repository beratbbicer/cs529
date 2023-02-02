import json
import numpy as np
import emoji

root = './twibot-20_filtered'
train_file = f'{root}/train-filtered.json'
test_file = f'{root}/test-filtered.json'

with open(train_file, 'r') as file:
    train_data = json.load(file)

with open(test_file, 'r') as file:
    test_data = json.load(file)

seperator = '|'
with open('nodes.csv', 'w') as file:
    file.write(f'ID{seperator}Name{seperator}Follower Count{seperator}Friends Count{seperator}Label\n')
    for key, value in {**train_data, **test_data}.items():
        new_key = emoji.replace_emoji(key).strip()
        name = emoji.replace_emoji(emoji.demojize(value[0])).strip()
        file.write(f'{new_key}{seperator}{name}{seperator}{int(value[1])}{seperator}{int(value[2])}{seperator}{int(value[5])}\n')

with open('edges.csv', 'w') as file:
    file.write(f'Source{seperator}Target\n')
    for key, value in {**train_data, **test_data}.items():
        new_key = emoji.replace_emoji(key).strip()
        if value[3] is not None:
            if value[3]['following'] is not None:
                for id in value[3]['following']:
                    file.write(f'{new_key}{seperator}{id}\n')
_ = 1