'''
    reading ondevice-baseline results from log
'''

def acc_from_log(path, num_user=400, num_epochs=25):
    init_accs = [0.0 for _ in range(num_user)]
    epochs_accs = [[0.0 for _ in range(num_user)] for _ in range(num_epochs)]
    test_sizes = [0 for _ in range(num_user)]
    with open(path, 'r') as f:
        for line in f:
            # Init Model, Client: x, Testset Size: x, valid acc: x
            if 'Init Model, Client' in line:
                client_num = int(line.split(',')[1].split(' ')[-1])
                acc = float(line.split(':')[-1].strip())
                init_accs[client_num] = acc
                # test_sizes[client_num] = int(line.split(',')[2].split(' ')[-1])
            # Epoch: x, Client: x, Testset Size: x, valid acc: x
            if 'Epoch' in line:
                epoch = int(line.split(',')[0].split(' ')[-1])
                client_num = int(line.split(',')[1].split(' ')[-1])
                acc = float(line.split(':')[-1].strip())
                epochs_accs[epoch][client_num] = acc
                test_sizes[client_num] = int(line.split(',')[2].split(' ')[-1])
    # init weighted averaged acc
    init_acc = sum([init_accs[i]*test_sizes[i] for i in range(num_user)]) / sum(test_sizes)
    print(f'Init: {init_acc}')
    # epoch weighted averaged acc
    for i in range(num_epochs):
        averaged_epoch_acc = sum([epochs_accs[i][j]*test_sizes[j] for j in range(num_user)]) / sum(test_sizes)
        print(f'Epoch {i}: {averaged_epoch_acc}')
    return init_accs, epochs_accs, test_sizes


def acc_from_log_unknown(path, num_epochs=25):
    '''
        total number of clients is unknown
    '''
    init_accs = {}
    epochs_accs = [{} for _ in range(num_epochs)]
    test_sizes = {}
    with open(path, 'r') as f:
        for line in f:
            # Init Model, Client: x, Testset Size: x, valid acc: x
            if 'Init Model, Client' in line:
                client_num = int(line.split(',')[1].split(' ')[-1])
                acc = float(line.split(':')[-1].strip())
                init_accs[client_num] = acc
                # test_sizes[client_num] = int(line.split(',')[2].split(' ')[-1])
            # Epoch: x, Client: x, Testset Size: x, valid acc: x
            if 'Epoch' in line:
                epoch = int(line.split(',')[0].split(' ')[-1])
                client_num = int(line.split(',')[1].split(' ')[-1])
                acc = float(line.split(':')[-1].strip())
                epochs_accs[epoch][client_num] = acc
                test_sizes[client_num] = int(line.split(',')[2].split(' ')[-1])
    # init weighted averaged acc
    print(len(init_accs), len(test_sizes))
    # assert len(init_accs) == len(test_sizes)

    total_data = sum([test_sizes[i] for i in test_sizes.keys()])
    init_acc = sum([init_accs[i]*test_sizes[i] for i in init_accs.keys() if i in test_sizes.keys()]) / total_data
    print(f'Init: {init_acc}')
    # epoch weighted averaged acc
    for i in range(num_epochs):
        averaged_epoch_acc = sum([epochs_accs[i][j]*test_sizes[j] for j in init_accs.keys() if j in epochs_accs[i].keys()]) / total_data
        print(f'Epoch {i}: {averaged_epoch_acc}')
    return init_accs, epochs_accs, test_sizes