# -*- coding: utf-8 -*-

"""
A pytorch implementation of DeepFM for rates prediction problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import table_batched_embeddings_ops
from torch.autograd.profiler import record_function
from torch.profiler import ExecutionGraphObserver
import tempfile

# import graph_observer
# from caffe2.python import core
# core.GlobalInit(
#     [
#         "python",
#         "--pytorch_enable_execution_graph_observer=true",
#         "--pytorch_execution_graph_observer_iter_label=## BENCHMARK ##",
#     ]
# )

from time import time

def _time(is_cuda):
    if is_cuda:
        torch.cuda.synchronize()

    return time()


class DeepFM(nn.Module):
    """
    A DeepFM network with RMSE loss for rates prediction problem.

    There are two parts in the architecture of this network: fm part for low
    order interactions of features and deep part for higher order. In this 
    network, we use bachnorm and dropout technology for all hidden layers,
    and "Adam" method for optimazation.

    You may find more details in this paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
    """

    def __init__(self, feature_sizes, embedding_size=4,
                 hidden_dims=[32, 32], num_classes=10, dropout=[0.5, 0.5], 
                 use_cuda=True, verbose=False):
        """
        Initialize a new network

        Inputs:
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dtype = torch.float

        """
            check if use cuda
        """
        self.use_cuda = False
        if use_cuda and torch.cuda.is_available():
            self.use_cuda = True
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        """
            init fm part
        """

        self.fm_first_order_Linears = nn.Conv1d(
                in_channels=self.feature_sizes[0] * 13,
                out_channels=self.embedding_size * 13,
                kernel_size=1,
                groups=13
        ) # "parallel multi-head mm", assuming EQUAL feature sizes for linear 
        self.fm_first_order_Embeddings = table_batched_embeddings_ops.TableBatchedEmbeddingBags(
            26,
            self.feature_sizes[13:],
            self.embedding_size,
            optimizer=table_batched_embeddings_ops.Optimizer.SGD,
            learning_rate=1e-9,
            eps=None,
            stochastic_rounding=False,
        )

        self.fm_second_order_Linears = nn.Conv1d(
                in_channels=self.feature_sizes[0] * 13,
                out_channels=self.embedding_size * 13,
                kernel_size=1,
                groups=13
        ) # "parallel multi-head mm", assuming EQUAL feature sizes for linear
        self.fm_second_order_Embeddings = table_batched_embeddings_ops.TableBatchedEmbeddingBags(
            26,
            self.feature_sizes[13:],
            self.embedding_size,
            optimizer=table_batched_embeddings_ops.Optimizer.SGD,
            learning_rate=1e-9,
            eps=None,
            stochastic_rounding=False,
        )

        """
            init deep part
        """
        all_dims = [self.field_size * self.embedding_size] + \
            self.hidden_dims + [self.num_classes]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_'+str(i),
                    nn.Linear(all_dims[i-1], all_dims[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
            setattr(self, 'batchNorm_' + str(i),
                    nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_'+str(i),
                    nn.Dropout(dropout[i-1]))

    def forward(self, Xi, Xv):
        """
        Forward process of network. 

        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        """
            fm part
        """
        # Non-batched: 
        # - embedding input per table: (B, L(=1))
        # - embedding output per table: (B, 1, D), summed to (B, D), 26 outputs in total
        # - Xv shape: (B, 39), Xv[:, i] shape: (B); element = (D, B) dot_mul (B) = (D, B)
        # - fm_first/second_order_emb_arr: len of 39, (B, D)
        # - fm_first_order: (B, 39 * D)
        # - fm_second_order: (B, D)
        # Batched:
        # - embedding input per table: (B, 26, L(=1))
        # - embedding output per table: (B, 26, D); Xv shape: (B, 39)
        # - fm_first_order_emb_arr: （B, 26, D), transposed to (26, D, B); 
        # - Xv shape: (B, 39), Xv[:, i:] shape (B, 26), transposed to (26, B)
        # - fm_first_order_linear: (B, 13 * D) -> (13 * B, D) -> (B, 13 * D)
        # - fm_first_order_emb: (B, 26 * D)
        # - fm_second_order_linear: (B, 13 * D) -> (B, 13, D)
        # - fm_second_order_emb: (B, 26 * D)
        # - fm_first_order: (B, 39 * D)
        # - fm_second_order: (B, D)

        Xi_linear = Xi[:, :13, :].to(device=self.device, dtype=torch.float)
        Xi_tem = Xi[:, 13:, :].to(device=self.device, dtype=torch.int32).view(-1)
        B, S, _ = Xi.shape
        S -= 13 # To avoid slicing
        offsets = torch.arange(0, B*S+1, dtype=torch.int32).cuda()

        ### First order
        fm_first_order_linear = self.fm_first_order_Linears(Xi_linear)
        fm_first_order_linear = torch.transpose(fm_first_order_linear, 0, 1).reshape(-1, self.embedding_size)
        fm_first_order_linear = torch.mul(fm_first_order_linear.view(Xi.shape[0], 13, self.embedding_size), Xv[:, :13].unsqueeze(-1)).view(B, -1)

        fm_first_order_emb = (self.fm_first_order_Embeddings(Xi_tem, offsets).view(S, -1, B) * Xv[:, 13:].reshape(S, 1, B)).view(B, -1)
        fm_first_order = torch.cat([fm_first_order_linear, fm_first_order_emb], dim=1)

        ### Second order: use 2xy = (x+y)^2 - x^2 - y^2 to reduce calculation
        fm_second_order_linear = self.fm_second_order_Linears(Xi_linear)
        fm_second_order_emb = (self.fm_second_order_Embeddings(Xi_tem, offsets).view(S, -1, B) * Xv[:, 13:].reshape(S, 1, B)).view(B, S, -1)

        """ deep part (executes here before transposing fm_second_order_linear) """
        deep_emb = torch.cat([fm_second_order_linear.view(B, -1), fm_second_order_emb.view(B, -1)], dim=1)
        deep_out = deep_emb
        """ deep part ends """

        fm_second_order_linear = torch.transpose(fm_second_order_linear.reshape(B, -1, self.embedding_size), 0, 1)
        fm_second_order_emb_arr = [fm_second_order_linear, fm_second_order_emb]
        
        fm_sum_second_order_emb = sum([torch.sum(x, 0) if idx != len(fm_second_order_emb_arr)-1 \
                                        else torch.sum(x, 1) \
                                        for idx, x in enumerate(fm_second_order_emb_arr)])
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * \
                                            fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum([torch.sum(x, 0) if idx != len(fm_second_order_emb_arr)-1 \
                                                else torch.sum(x, 1) \
                                                for idx, x in enumerate(fm_second_order_emb_square)])  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square - \
                           fm_second_order_emb_square_sum) * 0.5

        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
        """
            sum
        """
        bias = torch.nn.Parameter(torch.randn(Xi.size(0)))
        if self.use_cuda:
            bias = bias.cuda()
        total_sum = torch.sum(fm_first_order, dim=1) + \
                    torch.sum(fm_second_order, dim=1) + \
                    torch.sum(deep_out, dim=1) + bias
        return total_sum

    def fit(self, loader_train, loader_val, optimizer, batch_size=32, epochs=1, warmup=5, batch_limit=1e9, verbose=False, print_every=5, collect_execution_graph=False):
        """
        Training a model and valid accuracy.

        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations. 
        """
        """
            load input data
        """
        model = self.train().to(device=self.device)
        criterion = F.binary_cross_entropy_with_logits

        if collect_execution_graph:
            fp = tempfile.NamedTemporaryFile('w+t', prefix='/tmp/pytorch_execution_graph_', suffix='.json', delete=False)
            fp.close()
            eg = ExecutionGraphObserver()
            eg.register_callback(fp.name)

        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        time_fwd = 0
        time_bwd = 0
        global_batch_count = 0
        should_return = False

        # Warm up
        for t, (xi, xv, y) in enumerate(loader_train):
            xi = xi.to(device=self.device, dtype=self.dtype)
            xv = xv.to(device=self.device, dtype=torch.float)
            y = y.to(device=self.device, dtype=self.dtype)
            total = model(xi, xv)
            loss = criterion(total, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_batch_count += 1
            if global_batch_count >= warmup:
                global_batch_count = 0
                break

        for epoch in range(epochs):
            for t, (xi, xv, y) in enumerate(loader_train):
                if xi.shape[0] != batch_size:
                    continue
                if global_batch_count == 0 and collect_execution_graph:
                    eg.start()
                with record_function("## Forward ##"):
                    with record_function("module::forward_pass::transfer_gpu_data"):
                        xi = xi.to(device=self.device, dtype=self.dtype)
                        xv = xv.to(device=self.device, dtype=torch.float)
                        y = y.to(device=self.device, dtype=self.dtype)
                    t1 = _time(self.use_cuda)
                    if self.use_cuda:
                        event_start.record()
                    total = model(xi, xv)
                    if self.use_cuda:
                        event_end.record()
                    t2 = _time(self.use_cuda)
                time_fwd += event_start.elapsed_time(event_end) * 1.e-3 if self.use_cuda else (t2 - t1)
                with record_function("## Backward ##"):
                    t1 = _time(self.use_cuda)
                    if self.use_cuda:
                        event_start.record()
                    loss = criterion(total, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if self.use_cuda:
                        event_end.record()
                    t2 = _time(self.use_cuda)
                if global_batch_count == 0 and collect_execution_graph:
                    eg.stop()
                    eg.unregister_callback()
                time_bwd += event_start.elapsed_time(event_end) * 1.e-3 if self.use_cuda else (t2 - t1)

                if verbose and global_batch_count % print_every == 0:
                    print('Epoch %d Iteration %d, loss = %.4f' % (epoch, t, loss.item()))
                    self.check_accuracy(loader_val, model)
                global_batch_count += 1
                if global_batch_count >= batch_limit:
                    should_return = True
                    break
            if should_return:
                break

        time_fwd_avg = time_fwd / global_batch_count * 1000
        time_bwd_avg = time_bwd / global_batch_count * 1000
        time_total = time_fwd_avg + time_bwd_avg

        print("Overall per-batch training time: {:.2f} ms".format(time_total))

    def check_accuracy(self, loader, model):
        if loader.dataset.train:
            print('  Checking accuracy on validation set')
        else:
            print('  Checking accuracy on test set')
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for xi, xv, y in loader:
                xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                xv = xv.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)
                total = model(xi, xv)
                preds = (torch.sigmoid(total) > 0.5).to(dtype=self.dtype)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('  Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
