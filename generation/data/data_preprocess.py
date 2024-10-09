'''
    Pre-process the REDDIT dataset from LEAF benchmark
'''
import os
import json
import pickle


def find_users_reddit(num):
    user_list = []
    num_list = []
    for file_idx in range(21):
        print(f'Start to process file {file_idx}')
        filepath = f'./data/reddit_leaf/test/reddit_{file_idx}_test.json'
        data = json.load(open(filepath, 'r', encoding='utf8'))
        tmp_num_list = data['num_samples']
        for i in range(len(tmp_num_list)):
            if tmp_num_list[i] > num:
                num_list.append(tmp_num_list[i])
                user_list.append(data['users'][i])
        print(f'Now there are {len(num_list)} users')
    pickle.dump(user_list, open(f'./data/reddit/chosen_user_{num}_{len(num_list)}.pickle', 'wb'))


def generate_reddit_user_files_raw(path):
    user_list = pickle.load(open(path, 'rb'))
    datatypes = ['test','train']
    for datatype in datatypes:
        user_id = 1
        for file_idx in range(21):
            print(f'Start to process file {file_idx}')
            filepath = f'./data/reddit_leaf/{datatype}/reddit_{file_idx}_{datatype}.json'
            data = json.load(open(filepath, 'r', encoding='utf8'))
            for user in user_list:
                user = str(user)
                processed = []
                if user in data['users']:
                    print(f'Processing user {user}')
                    user_data = data['user_data'][user]
                    for comment_idx in range(len(user_data['x'])):
                        for sample_idx in range(len(user_data['x'][comment_idx])):
                            processed.append({"input":user_data['x'][comment_idx][sample_idx], "top1":user_data['y'][comment_idx]['target_tokens'][sample_idx]})
                    json.dump(processed, open(f'./data/reddit_leaf/1407_user/raw_data/{datatype}/user_{user_id}.json', 'w'), ensure_ascii=False)
                    user_id += 1
                else:
                    continue
        

def generate_reddit_user_files():
    for datatype in ['test', 'train']:
        for user in range(1, 1408):
            print(f'Start to process user {user}')
            filepath = f'./data/reddit_leaf/1407_user/raw_data/{datatype}/user_{user}.json'
            datas = json.load(open(filepath, 'r', encoding='utf8'))
            processed = []
            for data in datas:
                flag = False
                input = data['input']
                target = data['top1']
                for word in input:
                    if len(word) > 40:
                        print('Error data! deleted!', word)
                        flag = True
                        break
                if flag:
                    continue
                for i in range(len(input)):
                    if input[i] in ['<BOS>', '<PAD>', '<EOS>']:
                        continue
                    if target[i] in ['<PAD>', '<EOS>']:
                        continue
                    till_now = ' '.join(input[:i+1])
                    processed.append({"input":till_now, "top1":target[i]})
            json.dump(processed, open(f'./data/reddit_leaf/1407_user/user_data/{datatype}/user_{user}.json', 'w'), ensure_ascii=False)


def merge_data():
    for type in ['test', 'train']:
        source_path = f'./data/reddit_leaf/1407_user/user_data/{type}/'
        target_file = f'./data/reddit_leaf/1407_user/cloud_data/Reddit_client_{type}_data_new.json'
        with open(target_file, 'w', encoding='utf8') as f:
            for client_num in range(1, 1408):
                print(f'Client {client_num}')
                source_file = os.path.join(source_path, f'user_{client_num}.json')
                data = json.load(open(source_file, "r"))
                for sample in data:
                    line = {'input': sample['input'], 'top1':sample['top1'], 'client':client_num}
                    result = json.dumps(line, ensure_ascii=False) + '\n'
                    f.write(result)


if __name__ == '__main__':
    generate_reddit_user_files_raw('./data/reddit_leaf/chosen_user_300_310_1407.pickle')
    generate_reddit_user_files()
    merge_data()