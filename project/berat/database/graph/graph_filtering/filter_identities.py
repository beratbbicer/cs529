import json

dbroot = '/media/HDD/berat/TwiBot-20/database'

dev_file = f'{dbroot}/dev.json'
dev_file_filtered = './dev-filtered.json'

with open(dev_file, 'r') as file:
    table = json.load(file)

dev_dict = {}
for entry in table:
    if 'Politics' in entry['domain']:
        dev_dict[entry['ID']] = [
            entry['profile']['name'],
            entry['profile']['followers_count'],
            entry['profile']['friends_count'],
            entry['neighbor'],
            None, # entry['tweet'],
            entry['label']
        ]

with open(dev_file_filtered, 'w') as file:
    json.dump(dev_dict, file)

############################################

support_file = f'{dbroot}/support.json'
support_file_filtered = './support-filtered.json'

with open(support_file, 'r') as file:
    table = json.load(file)

support_dict = {}
for entry in table:
    if 'Politics' in entry['domain']:
        support_dict[entry['ID']] = [
            entry['profile']['name'],
            entry['profile']['followers_count'],
            entry['profile']['friends_count'],
            entry['neighbor'],
            None, # entry['tweet'],
            None # no bot label
        ]

with open(support_file_filtered, 'w') as file:
    json.dump(support_dict, file)

############################################

train_file = f'{dbroot}/train.json'
train_file_filtered = './train-filtered.json'

with open(train_file, 'r') as file:
    table = json.load(file)

train_dict = {}
for entry in table:
    if 'Politics' in entry['domain']:
        train_dict[entry['ID']] = [
            entry['profile']['name'],
            entry['profile']['followers_count'],
            entry['profile']['friends_count'],
            entry['neighbor'],
            None, # entry['tweet'],
            entry['label']
        ]

with open(train_file_filtered, 'w') as file:
    json.dump(train_dict, file)

############################################

test_file = f'{dbroot}/test.json'
test_file_filtered = './test-filtered.json'

with open(test_file, 'r') as file:
    table = json.load(file)

test_dict = {}
for entry in table:
    if 'Politics' in entry['domain']:
        test_dict[entry['ID']] = [
            entry['profile']['name'],
            entry['profile']['followers_count'],
            entry['profile']['friends_count'],
            entry['neighbor'],
            None, # entry['tweet'],
            entry['label']
        ]

with open(test_file_filtered, 'w') as file:
    json.dump(test_dict, file)