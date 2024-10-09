import json
from datetime import datetime


def senti_data_per_user():
    file = 'training.1600000.processed.noemoticon.csv'
    f = open(file, 'r', encoding='latin-1')
    data = f.readlines()
    user_counter = {}
    for line in data:
        # print(line)
        # exit()
        words = line.split(',')
        user = words[4]
        user_counter[user] = user_counter.get(user, 0) + 1
    pairs = [(user, user_counter[user]) for user in user_counter.keys()]
    decend_pair = sorted(pairs, key=lambda x:x[1], reverse=True)
    # print(decend_pair[:163])
    user_dict = set()
    for (sel_user, _) in decend_pair[:163]:
        user_dict.add(sel_user)
    return user_dict


def user_data(users):
    file = 'training.1600000.processed.noemoticon.csv'
    f = open(file, 'r', encoding='latin-1')
    data = f.readlines()
    filtered_data = []
    for line in data:
        # print(line)
        # exit()
        words = line.split(',')
        user = words[4]
        label = words[0]
        time = words[2]
        review = words[5]
        if user in users:
            data_dict = {
                'user': user,
                'label': label,
                'time': time,
                'review': review
            }
            filtered_data.append(data_dict)
    with open('filtered_100_alldata.json', 'w') as json_file:
        json.dump(filtered_data, json_file, ensure_ascii=False, indent=4)


def time_split():
    file_path = 'filtered_100_alldata.json'
    time_format = '%a %b %d %H:%M:%S PDT %Y'
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    times = []
    for sample in data:
        time = sample['time']
        time = time.replace('"', '')
        time = datetime.strptime(time, time_format)
        times.append(time)
    sort_time = sorted(times)
    n_sample = len(sort_time)
    train_sample = int(0.8*n_sample)
    print(sort_time[train_sample])
    print(f'training sample: {train_sample}, test sample: {n_sample-train_sample}')


def central_data(time_step='2009-06-06 23:53:52'):
    file_path = 'filtered_100_alldata.json'
    time_format = '%a %b %d %H:%M:%S PDT %Y'
    time_step_dt = datetime.strptime(time_step, '%Y-%m-%d %H:%M:%S')
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    train_data, test_data = [], []
    for sample in data:
        time = sample['time']
        time = time.replace('"', '')
        sample_time_dt = datetime.strptime(time, time_format)
        if sample_time_dt < time_step_dt:
            train_data.append(sample)
        else:
            test_data.append(sample)
    print(len(train_data), len(test_data))
    with open('filtered100_central_train.json', 'w') as json_file:
        json.dump(train_data, json_file, ensure_ascii=False, indent=4)
    with open('filtered100_central_test.json', 'w') as json_file:
        json.dump(test_data, json_file, ensure_ascii=False, indent=4)       


def client_central_data(path):
    train_path = f'user_train/'
    test_path = f'user_test/'

    train_data = []
    for i in range(1, 165):
        filename = f'{train_path}user_{i}.json'
        # print(filename)
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                for sample in data:
                    # print(sample)
                    sample['user'] = i
                    train_data.append(sample)
        except:
            pass
    test_data = []
    for i in range(1, 165):
        filename = f'{test_path}user_{i}.json'
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                for sample in data:
                    sample['user'] = i
                    test_data.append(sample)
        except:
            pass

    print(len(train_data), len(test_data))
    with open('filtered100_client_train.json', 'w') as json_file:
        json.dump(train_data, json_file, ensure_ascii=False, indent=4)
    with open('filtered100_client_test.json', 'w') as json_file:
        json.dump(test_data, json_file, ensure_ascii=False, indent=4)                  


def data_sample():
    with open('./data/sentiment_data/filtered100_client_test.json', 'r') as file:
        test_data = json.load(file)   

    import random
    sample_test_data = []
    sampled_ids = random.sample(range(1, len(test_data)), 100)

    for idx in sampled_ids:
        sample_test_data.append(test_data[idx])
    
    test_path = f'./data/sentiment_data/sample_test.json'
    with open(test_path, 'w') as json_file:
        json.dump(sample_test_data, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # users = senti_data_per_user()
    # user_data(users)
    # time_split()
    # central_data()
    # client_central_data('.')
    data_sample()