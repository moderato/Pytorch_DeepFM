import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.autograd.profiler as profiler
import argparse

from model.DeepFM import DeepFM
from data.dataset import CriteoDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DeepFM Benchmark')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                       help='disable CUDA')
    parser.add_argument('--profile', action='store_true', default=False,
                       help='enable autograd profiler')
    parser.add_argument('--collect-execution-graph', action='store_true', default=False,
                       help='collect execution graph')
    parser.add_argument("--batch-size", type=int, default=64,
                       help='batch size')
    parser.add_argument("--num-epoch", type=int, default=20,
                       help='nb of epochs in loop to average perf')
    parser.add_argument("--num-batches", type=int, default=1e9,
                       help='nb of batches in loop to average perf')
    parser.add_argument("--print-freq", type=int, default=5,
                       help='print frequency')
    parser.add_argument("--engine-type", type=str, default='gmf',
                       help='nb of batches in loop to average perf')
    parser.add_argument('--evaluate', action='store_true', default=False,
                       help='evaluate after training')
    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    # load data
    train_data = CriteoDataset('./data', train=True)
    loader_train = DataLoader(train_data, batch_size=args.batch_size,
                            sampler=sampler.SubsetRandomSampler(range(800)))
    val_data = CriteoDataset('./data', train=True)
    loader_val = DataLoader(val_data, batch_size=args.batch_size,
                            sampler=sampler.SubsetRandomSampler(range(800, 899)))

    feature_sizes = np.loadtxt('./data/feature_sizes.txt', delimiter=',')
    feature_sizes = [int(x) for x in feature_sizes]
    # print(feature_sizes)

    model = DeepFM(feature_sizes, use_cuda=args.cuda)
    optimizer = optim.SGD(model.parameters(), lr=1e-9, momentum=0.9)
    with profiler.profile(args.profile, use_cuda=args.cuda, use_kineto=True) as prof:
        model.fit(loader_train, loader_val, optimizer, \
                    epochs=args.num_epoch, \
                    verbose=True, \
                    batch_limit=args.num_batches, \
                    print_every=args.print_freq, \
                    collect_execution_graph=args.collect_execution_graph)

    if args.profile:
        with open("deepfm_benchmark.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
        prof.export_chrome_trace("deepfm_benchmark.json")
