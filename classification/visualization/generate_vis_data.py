'''
    Pre-process the sampled data which is used for visulization
'''
import json
import math
import pickle
import random
import bisect


class VisData:
    def __init__(self, prompt, label) -> None:
        self.prompt = prompt
        self.label = label
        self.counter = [0, 0, 0]

    def __repr__(self) -> str:
        return f'{self.prompt}\nLabel: {self.label}, \tBaseline prediction: {self.baseline}, \tOur prediction: {self.prediction}, \tH: {self.H}\nRandom: {self.counter}'

    def set_prediction(self, prompt, label):
        assert prompt == self.prompt
        self.prediction = label
    
    def set_baseline(self, prompt, label):
        assert prompt == self.prompt
        self.baseline = label

    def set_random(self, prompt, label):
        assert prompt == self.prompt
        self.random = label

    def set_counter(self, prompt, labels):
        assert prompt == self.prompt
        for pred in labels:
            self.counter[pred] += 1

    def compute_H(self):
        random_times = sum(self.counter)
        assert random_times > 0
        possibility = [i / random_times for i in self.counter]
        self.H = 0.
        for pos in possibility:
            if pos > 0:
                self.H += (-pos * math.log(pos))


class Visualize:
    def __init__(self, num = 0) -> None:
        self.num = num
        self.data = []

    def from_json(self, baseline, ours):
        assert self.data == []
        with open(baseline, 'r') as f:
            baseline_lines = f.readlines()
        with open(ours, 'r') as f:
            ours_lines = f.readlines()
        for i in range(self.num):
            ann_base = json.loads(baseline_lines[i])
            ann_ours = json.loads(ours_lines[i])
            vis_data = VisData(ann_base['input'], ann_base['label'])
            vis_data.set_prediction(ann_ours['input'], ann_ours['pred'])
            vis_data.set_baseline(ann_base['input'], ann_base['pred'])
            vis_data.set_counter(ann_ours['input'], ann_ours['random'])
            vis_data.set_random(ann_ours['input'], ann_ours['random'][random.randint(0, len(ann_ours['random']) - 1)])
            vis_data.compute_H()
            self.data.append(vis_data)

    def from_pickle(self, path):
        assert self.data == []
        self.data = pickle.load(open(path, 'rb'))
        self.num = len(self.data)

    def save_data(self, path):
        assert self.data != []
        pickle.dump(self.data, open(path, 'wb'))

    def _sort_data_by_H(self, reverse=False):
        assert len(self.data) > 0
        self.data.sort(key=lambda x:x.H, reverse=reverse)
    
    def _compute_acc(self, shard):
        right_ours = [data.label == data.prediction for data in shard]
        right_baseline = [data.label == data.baseline for data in shard]
        if hasattr(shard[0], 'random'):
            right_random = [data.label == data.random for data in shard]
            return sum(right_ours)/len(shard), sum(right_baseline)/len(shard), sum(right_random)/len(shard)
        return sum(right_ours)/len(shard), sum(right_baseline)/len(shard)
        
    def compute_acc_all(self):
        acc_our, acc_base, acc_rand = self._compute_acc(self.data)
        print(f'All data:\nAcc baseline:{acc_base:.4f}, Acc random:{acc_rand:.4f}, Acc ours:{acc_our:.4f}')

    def split_by_step(self, stepsize):
        self._sort_data_by_H()
        splited_data = [self.data[i*stepsize:(i+1)*stepsize] for i in range(self.num // stepsize)]
        splited_data.append(self.data[(self.num // stepsize) * stepsize:])
        acc_list = []
        for shard in splited_data:
            acc_our, acc_base, acc_rand = self._compute_acc(shard)
            acc_list.append([acc_our, acc_base, acc_rand])
            print('='*50)
            print(f'{stepsize} points, H range from {shard[0].H} to {shard[-1].H}:')
            print(f'Acc baseline:{acc_base:.4f}, Acc random:{acc_rand:.4f}, Acc ours:{acc_our:.4f}')
        return splited_data, acc_list
    
    def split_by_H(self, shard_num):
        self._sort_data_by_H()
        ceil_H = (int(self.data[-1].H * 100) + 1) / 100
        step_width = ceil_H / shard_num
        Hs = [data.H for data in self.data]
        split_point = [0] + [bisect.bisect(Hs, step_width * i) for i in range(1, shard_num)]
        splited_data_tail = [self.data[split_point[-1]:]]
        splited_data_mid = [self.data[split_point[i]:split_point[i+1]] for i in range(shard_num - 1)]
        splited_data = splited_data_mid +splited_data_tail
        acc_list = []
        for i in range(len(splited_data)):
            shard = splited_data[i]
            if i == len(splited_data) - 1:
                data_points = self.num - split_point[-1]
            else:
                data_points = split_point[i + 1] - split_point[i]
            acc_our, acc_base, acc_rand = self._compute_acc(shard)
            acc_list.append([acc_our, acc_base, acc_rand])
            print('='*50)
            print(f'{data_points} points, H range from {shard[0].H} to {shard[-1].H}:')
            print(f'Acc baseline:{acc_base:.4f}, Acc random:{acc_rand:.4f}, Acc ours:{acc_our:.4f}')
        return acc_list, step_width

    def check_H(self):
        for data in self.data:
            if data.H == 0.:
                if data.random != data.prediction:
                    print(data)
                    exit()

    def find_predict_higher(self):
        collects = []
        for data in self.data:
            if data.baseline == 1 and (data.label==0 and data.prediction==0):
                collects.append(data)
        return collects


if __name__ == '__main__':
    visual = Visualize(10000)
    visual.from_json('', '')
    # visual.save_data('./final_results/amazonBART_data_beta.pickle')
    # visual.from_pickle('./amazon_data.pickle')
    # for data in visual.data:
    #     print(data)
    # visual.check_H()
    visual.compute_acc_all()
    visual.split_by_H(6)
    c = visual.find_predict_higher()
    print('End')
