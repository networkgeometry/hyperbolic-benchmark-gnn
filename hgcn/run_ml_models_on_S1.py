import os
import textwrap
import sys

"""
We consider the following models:

1. MLP (Multi-Layer Perceptron)
2. HNN (Hyperbolic Neural Network)
3. GCN (Graph Convolutional Neural Networks)
4. GAT (Graph Attention Networks)
5. HGCN (Hyperbolic Graph Convolutional Neural Networks)
"""

def compose_train_command(task, dataset_path, config, nc_test_size=0.7):
    return textwrap.dedent(f"""
        python train.py \
            --task {task} \
            --dataset S1 \
            --data_path_S1 {dataset_path} \
            --model {config['model']} \
            --lr 0.01 \
            --dim 16 \
            --num-layers {config['num-layers']} \
            --act {config['act']} \
            --bias {config['bias']} \
            --dropout {config['dropout']} \
            --weight-decay {config['weight-decay']} \
            --manifold {config['manifold']} \
            --use-feats {config['use-feats']} \
            --log-freq 5 \
            --cuda 0 \
            --c None \
            --ntimes 5 \
            --test-size {nc_test_size} > results/output_task_{task}_model_{config['model']}_{config['manifold']}_with_features_{config['use-feats']}_nc_testsize_{nc_test_size}_{os.path.basename(dataset_path)}.log
    """)


def get_config_per_model(model):
    if model == 'MLP':
        config = {
            'model': 'MLP',
            'manifold': 'Euclidean',
            'dropout': 0.2,
            'weight-decay': 0.001,
            'use-feats': 1,
            'optimizer': 'Adam',
            'bias': 0,
            'act': 'None',
            'num-layers': 2
        }
    elif model == 'HNN':
        config = {
            'model': 'HNN',
            'manifold': 'PoincareBall',
            'dropout': 0.2,
            'weight-decay': 0.001,
            'use-feats': 1,
            'optimizer': 'Adam',
            'bias': 1,
            'act': 'None',
            'num-layers': 2
        }
    elif model == 'GCN':
        config = {
            'model': 'GCN',
            'manifold': 'Euclidean',
            'dropout': 0.2,
            'weight-decay': 0.0005,
            'use-feats': 1,
            'optimizer': 'Adam',
            'bias': 1,
            'act': 'relu',
            'num-layers': 2
        }
    elif model == 'GAT':
        config = {
            'model': 'GAT',
            'manifold': 'Euclidean',
            'dropout': 0.2,
            'weight-decay': 0.0005,
            'use-feats': 1,
            'optimizer': 'Adam',
            'bias': 1,
            'act': 'relu',
            'num-layers': 2
        }
    elif model == 'HGCN':
        config = {
            'model': 'HGCN',
            'manifold': 'PoincareBall',
            'dropout': 0.2,
            'weight-decay': 0.0005,
            'use-feats': 1,
            'optimizer': 'Adam',
            'bias': 1,
            'act': 'relu',
            'num-layers': 2
        }
    return config
    

def run_parallel_commands(commands):
    with open('commands_tmp.txt', 'w') as f:
        for c in commands:
            f.write(f'{c.strip()}\n')
    os.system('parallel -j4 --delay 1 :::: commands_tmp.txt')
    os.system('rm commands_tmp.txt')


def run_single_commands(commands):
    for c in commands:
        print(c)
        os.system(c)


if __name__ == '__main__':
    task = sys.argv[1]
    model = sys.argv[2]

    net_folders = [f.path for f in os.scandir('./data') if f.is_dir() and 'output' in f.path]
    os.makedirs("results", exist_ok=True)

    config = get_config_per_model(model)
    run_commands = [compose_train_command(task, net, config) for net in net_folders]
    run_parallel_commands(run_commands)
    
