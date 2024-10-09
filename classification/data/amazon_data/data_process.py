import json


def time_split():
    data_folder = 'users/'
    times = []
    for user_num in range(1434):
        data_path = f'{data_folder}user{user_num}.json'
        f = open(data_path, 'r')
        for line in f:
            data_sample = json.loads(line)
            time = data_sample["time"]
            times.append(time)
    sort_time = sorted(times)
    n_sample = len(sort_time)
    train_sample = int(0.8*n_sample)
    print(sort_time[train_sample])
    print(f'training sample: {train_sample}, test sample: {n_sample-train_sample}')


def user_data():
    data_folder = 'users/'
    for user_num in range(1434, 1435):
        user_train, user_test = [], []
        data_path = f'{data_folder}user{user_num}.json'
        f = open(data_path, 'r')
        for line in f:
            data_sample = json.loads(line)
            rate = data_sample["rate"]
            text = data_sample["text"]
            time = data_sample["time"]
            user_id = user_num + 1
            data_dict = {
                'user': user_id,
                'label': rate,
                'time': time,
                'review': text
            }
            if time <= 1632712835970:
                user_train.append(data_dict)
            else:
                user_test.append(data_dict)
        if len(user_train) > 0:
            with open(f'user_train/user{user_num+1}.json', 'w') as json_file:
                json.dump(user_train, json_file, ensure_ascii=False, indent=4)
            if len(user_test) > 0:
                with open(f'user_test/user{user_num+1}.json', 'w') as json_file:
                    json.dump(user_test, json_file, ensure_ascii=False, indent=4)


def central_data():
    train_path = f'user_train/'
    test_path = f'user_test/'

    train_data = []
    for i in range(1435):
        filename = f'{train_path}user{i}.json'
        # print(filename)
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                for sample in data:
                    train_data.append(sample)
        except:
            pass

    test_data = []
    for i in range(1435):
        filename = f'{test_path}user{i}.json'
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                for sample in data:
                    test_data.append(sample)
        except:
            pass

    print(len(train_data), len(test_data))
    with open('client_train.json', 'w') as json_file:
        json.dump(train_data, json_file, ensure_ascii=False, indent=4)
    with open('client_test.json', 'w') as json_file:
        json.dump(test_data, json_file, ensure_ascii=False, indent=4)         


def rating_stat():
    data_path = '../data/amazon_data/client_test.json'
    ratings_cnt = [0 for _ in range(5)]
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
        for sample in data:
            rating = int(sample['label'])
            ratings_cnt[rating-1] += 1
    # print(ratings_cnt)
    ratio = [rating_cnt/sum(ratings_cnt) for rating_cnt in ratings_cnt]
    print(ratio)


def data_sample():
    with open('../data/amazon_data/client_test.json', 'r') as file:
        test_data = json.load(file)   

    import random
    sample_test_data = []
    sampled_ids = random.sample(range(1, len(test_data)), 100)

    for idx in sampled_ids:
        sample_test_data.append(test_data[idx])
    
    test_path = f'./data/amazon_data/sample_test.json'
    
    with open(test_path, 'w') as json_file:
        json.dump(sample_test_data, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # time_split()
    # central_data()
    # rating_stat()
    data_sample()