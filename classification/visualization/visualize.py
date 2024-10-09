import os
import math
import bisect
import matplotlib.pyplot as plt
from generate_vis_data import Visualize, VisData


class Painter:
    def __init__(self, data) -> None:
        if len(data) == 1:
            self.vis_data = data[0]
        else:
            self.vis_data = data

    def plot_bar(self, col, save_path, baseline='random'):
        step_size = math.ceil(self.vis_data.num / col)
        
        splited_data, acc_list = self.vis_data.split_by_step(step_size)
        
        min_entropies = [shard[0].H for shard in splited_data]
        max_entropies = [shard[-1].H for shard in splited_data]
        mean_accuracies = [acc[0] for acc in acc_list]
        
        start_coloum = bisect.bisect(max_entropies, 0.)
        
        min_entropies = min_entropies[start_coloum:]
        max_entropies = max_entropies[start_coloum:]
        mean_accuracies = mean_accuracies[start_coloum:]
        acc_list = acc_list[start_coloum:]
        
        x_ticks = [0] + max_entropies

        if baseline == 'random':
            mean_acc_base = [acc[2] for acc in acc_list]
        elif baseline == 'base':
            mean_acc_base = [acc[1] for acc in acc_list]
        else:
            raise NotImplementedError(f'Baseline {baseline} not implemented')
        
        widths = [i - j for (i, j) in zip(max_entropies, min_entropies)]
        fig, ax = plt.subplots(dpi=300)
        ax.set_xticks(x_ticks)
        rec_our = ax.bar(min_entropies, mean_accuracies, width=widths, align='edge', alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
        rec_base = ax.bar(min_entropies, mean_acc_base, width=widths, align='edge', alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
        ax.bar_label(rec_our)
        ax.bar_label(rec_base)
        ax.set_title(f'Our method vs {baseline}')
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Accuracy (%)')
        plt.savefig(save_path)
        plt.show()

    def plot_bar_even(self, col, save_path, baseline='random'):
        acc_list, step_width = self.vis_data.split_by_H(col)
        mean_accuracies = [round(acc[0], 2) for acc in acc_list]        
        x_ticks = [step_width * i for i in range(col + 1)]
        if baseline == 'random':
            mean_acc_base = [round(acc[2], 2) for acc in acc_list]
        elif baseline == 'base':
            mean_acc_base = [round(acc[1], 2) for acc in acc_list]
        else:
            raise NotImplementedError(f'Baseline {baseline} not implemented')
        fig, ax = plt.subplots(dpi=300)
        ax.set_xticks(x_ticks)
        rec_our = ax.bar(x_ticks[:-1], mean_accuracies, width=step_width, align='edge', alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
        rec_base = ax.bar(x_ticks[:-1], mean_acc_base, width=step_width, align='edge', alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
        ax.bar_label(rec_our)
        ax.bar_label(rec_base)
        ax.set_title(f'Our method vs {baseline}')
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Accuracy (%)')
        plt.savefig(save_path)
        plt.show()

    def plot_scatter(self, col, save_path, baseline='random'):
        assert self.vis_data.num % col == 0
        step_size = self.vis_data.num // col
        
        splited_data, acc_list = self.vis_data.split_by_step(step_size)
        
        min_entropies = [shard[0].H for shard in splited_data]
        max_entropies = [shard[-1].H for shard in splited_data]
        mean_accuracies = [acc[0] for acc in acc_list]
        
        start_coloum = bisect.bisect(max_entropies, 0.)
        
        min_entropies = min_entropies[start_coloum:]
        max_entropies = max_entropies[start_coloum:]
        mean_accuracies = mean_accuracies[start_coloum:]
        acc_list = acc_list[start_coloum:]
  
        if baseline == 'random':
            mean_acc_base = [acc[2] for acc in acc_list]
        elif baseline == 'base':
            mean_acc_base = [acc[1] for acc in acc_list]
        else:
            raise NotImplementedError(f'Baseline {baseline} not implemented')
        fig, ax = plt.subplots()
        
        mean_entropies = [(i + j) / 2 for (i, j) in zip(max_entropies, min_entropies)]
        ax.scatter(mean_entropies, mean_accuracies, color='blue')
        ax.scatter(mean_entropies, mean_acc_base, color='red')
        ax.set_title(f'Our method vs {baseline}')
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Accuracy (%)')
        plt.savefig(save_path)
        plt.show()

    def plot_multi_lines(self, col, save_path, dataset, model):
        fig, ax = plt.subplots(dpi=300)
        
        acc_list, step_width = self.vis_data[0].split_by_H(col)
        mean_accuracies = [round(acc[0], 2)*100 for acc in acc_list]        
        x_ticks = [step_width * i + step_width * 0.5 for i in range(col + 1)]
        plot_xticks = [0.00, 0.18, 0.36, 0.54, 0.72, 0.90, 1.10]
        mean_acc_base = [round(acc[1], 2)*100 for acc in acc_list]
        ax.plot(x_ticks[:-1], mean_accuracies, label='Gaussian Design', marker='o', linestyle='--', color=(217/255.0,130/255.0,129/255.0))

        print(x_ticks)

        acc_list, step_width = self.vis_data[1].split_by_H(col)
        mean_accuracies = [round(acc[0], 2)*100 for acc in acc_list]        
        x_ticks = [step_width * i + step_width * 0.5 for i in range(col + 1)]
        mean_acc_base_beta = [round(acc[1], 2)*100 for acc in acc_list]
        # assert mean_acc_base == mean_acc_base_beta
        ax.plot(x_ticks[:-1], mean_accuracies, label='Beta Design', marker='o', color=(144/255.0,195/255.0,135/255.0))
        
        print(x_ticks)

        rec_base = ax.plot(x_ticks[:-1], mean_acc_base, label='Central (Gaussian)', marker='o', linestyle='--', color=(131/255.0,172/255.0,206/255.0))
        rec_base_beta = ax.plot(x_ticks[:-1], mean_acc_base_beta, label='Central (Beta)', marker='o', color=(244/255.0,180/255.0,121/255.0))
        
        # rec_rand = ax.plot(x_ticks[:-1], mean_acc_rand, label='Random baseline', marker='.', linestyle='--')
        # rec_rand_beta = ax.plot(x_ticks[:-1], mean_acc_rand_beta, label='Random baseline beta', marker='.',linestyle='--')
        ax.set_xticks(plot_xticks)
        
        ax.set_ylim(50, 100)
        # ax.grid(color='grey', linestyle='-', alpha=0.3)
        # ax.set_title(f'{dataset}, {model}')
        ax.set_xlabel('Entropy w.r.t. Users', fontsize=23)
        ax.set_ylabel('Inference Accuracy (%)', fontsize=23)
        plt.tight_layout()
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        # plt.legend(loc='upper right')
        plt.savefig(save_path)
        # plt.show()

    def plot_legend(self, col, save_path):
        fig, ax = plt.subplots(dpi=300)
        
        acc_list, step_width = self.vis_data[0].split_by_H(col)
        mean_accuracies = [round(acc[0], 2)*100 for acc in acc_list]        
        x_ticks = [step_width * i + step_width * 0.5 for i in range(col + 1)]
        plot_xticks = [float(format(step_width * i, ".2f")) for i in range(col + 1)]
        mean_acc_base = [round(acc[1], 2)*100 for acc in acc_list]
        ax.plot(x_ticks[:-1], mean_accuracies, label='Design (Gaussian)', marker='o', linestyle='--', color=(217/255.0,130/255.0,129/255.0))

        acc_list, step_width = self.vis_data[1].split_by_H(col)
        mean_accuracies = [round(acc[0], 2)*100 for acc in acc_list]        
        x_ticks = [step_width * i + step_width * 0.5 for i in range(col + 1)]
        mean_acc_base_beta = [round(acc[1], 2)*100 for acc in acc_list]
        # assert mean_acc_base == mean_acc_base_beta
        ax.plot(x_ticks[:-1], mean_accuracies, label='Design (Beta)', marker='o', color=(144/255.0,195/255.0,135/255.0))
        
        rec_base = ax.plot(x_ticks[:-1], mean_acc_base, label='Central (Gaussian)', marker='o', linestyle='--', color=(131/255.0,172/255.0,206/255.0))
        rec_base_beta = ax.plot(x_ticks[:-1], mean_acc_base_beta, label='Central (Beta)', marker='o', color=(244/255.0,180/255.0,121/255.0))
        
        # rec_rand = ax.plot(x_ticks[:-1], mean_acc_rand, label='Random baseline', marker='.', linestyle='--')
        # rec_rand_beta = ax.plot(x_ticks[:-1], mean_acc_rand_beta, label='Random baseline beta', marker='.',linestyle='--')
        ax.set_xticks(plot_xticks)
        
        ax.set_ylim(40, 100)
        # ax.grid(color='grey', linestyle='-', alpha=0.3)
        # ax.set_title(f'{dataset}, {model}')
        ax.set_xlabel('Entropy w.r.t. Users', fontsize=23)
        ax.set_ylabel('Inference Accuracy (%)', fontsize=23)
        plt.tight_layout()
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        plt.legend(loc='upper right', ncols=1)
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_increase(self, col, save_path, dataset, model):
        fig, ax = plt.subplots(figsize=(5,2), dpi=300)
        
        acc_list, step_width = self.vis_data[0].split_by_H(col)
        mean_accuracies_gaussian = [round(acc[0], 2) for acc in acc_list]        
        x_ticks = [step_width * i + step_width * 0.5 for i in range(col + 1)]
        plot_xticks = [float(format(step_width * i, ".2f")) for i in range(col + 1)]
        mean_acc_rand = [round(acc[2], 2) for acc in acc_list]
        mean_acc_base = [round(acc[1], 2) for acc in acc_list]

        acc_list, step_width = self.vis_data[1].split_by_H(col)
        mean_accuracies_beta = [round(acc[0], 2) for acc in acc_list]        
        x_ticks = [step_width * i + step_width * 0.5 for i in range(col + 1)]
        mean_acc_rand_beta = [round(acc[2], 2) for acc in acc_list]
        mean_acc_base_beta = [round(acc[1], 2) for acc in acc_list]
        
        delta_gaussian = [(mean_accuracies_gaussian[i]-mean_acc_base[i])*100 for i in range(len(mean_accuracies_gaussian))]
        delta_beta = [(mean_accuracies_beta[i]-mean_acc_base_beta[i])*100 for i in range(len(mean_acc_base_beta))]
        rec_base = ax.plot(x_ticks[:-1], delta_gaussian, label='Central (Gaussian)', marker='^', linestyle='--', color=(217/255.0,130/255.0,129/255.0))
        rec_base_beta = ax.plot(x_ticks[:-1], delta_beta, label='Central (Beta)', marker='^', color=(144/255.0,195/255.0,135/255.0))
        
        # rec_rand = ax.plot(x_ticks[:-1], mean_acc_rand, label='Random baseline', marker='.', linestyle='--')
        # rec_rand_beta = ax.plot(x_ticks[:-1], mean_acc_rand_beta, label='Random baseline beta', marker='.',linestyle='--')
        # ax.set_xticks(plot_xticks)
        
        # ax.set_ylim(0.45, 1.0)
        # ax.grid(color='grey', linestyle='-', alpha=0.3)
        # ax.set_title(f'{dataset}, {model}')
        # ax.set_xlabel('Entropy', fontsize=23)
        # ax.set_ylabel('Accuracy (%)', fontsize=23)
        plt.tight_layout()
        # ax.tick_params(axis='x', labelsize=15)
        # ax.tick_params(axis='y', labelsize=15)
        ax.set_xticks([])
        ax.set_yticks([0, 10, 20])
        # plt.legend(loc='upper right')
        # plt.savefig(save_path)
        plt.show()

if __name__ == '__main__':
    dataset = 'amazon'
    model = 'GPT'
    visdata = Visualize()
    visdata.from_pickle(f'/{dataset}{model}_data.pickle')
    visdata_beta = Visualize()
    visdata_beta.from_pickle(f'/{dataset}{model}_data_beta.pickle')
    painter = Painter([visdata, visdata_beta])
    model_type = f'{dataset}/{model}'
    os.makedirs(f'./figures_1/{model_type}/', exist_ok=True)
    # painter.plot_multi_lines(6, f'./figures_1/debug.pdf', dataset, model)
    painter.plot_multi_lines(6, f'./figures_1/{model_type}/fig_{dataset}_{model}.pdf', dataset, model)
    # painter.plot_increase(6, f'./figures_1/{model_type}/subfig_{dataset}_{model}.pdf', dataset, model)
    # painter.plot_legend(6, './figures_1/legend_col.pdf')