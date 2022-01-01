
import argparse
from collections import defaultdict
from sklearn.datasets import load_svmlight_file
from math import log

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as data_utils
import random
import numpy as np
import matplotlib.pyplot as plt

DATASETS=("abalone" ,"bodyfat", "cpusmall", "housing", "mg", "mpg", "space_ga")
DATASET_FEATURES = {
    "abalone": 8,
    "bodyfat": 14,
    "cpusmall": 12,
    "housing": 13,
    "mg": 6,
    "mpg": 7,
    "space_ga": 6,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare (regularized) linear regresssion "\
        "training of mult models vs. single model")

    parser.add_argument("-s", "--seed", type=int, default=None,
                        help='random seed')
    parser.add_argument("-w", "--num_workers", type=int, default=0,
                        help="number of data workers")
    parser.add_argument("--cuda", action="store_true",
                        help="whether to use cuda or not")

    parser.add_argument("--dataset", type=str, choices=DATASETS, default="space_ga",
                        help="name of dataset (expected to be in 'datasets' directory")
    parser.add_argument("--feature_dim", type=int, default=None,
                        help="features dimension (default: infer from dataset)")
    parser.add_argument("--output_dim", type=int, default=1,
                        help="output dimension (default: 1)")

    parser.add_argument("-M", "--num_models", dest="M", type=int, default=10,
                        help="number of multiple models")
    parser.add_argument("-T", "--num_iters", dest="T", type=int, default=100,
                        help="number of iterations")
    parser.add_argument("--agg_shots", type=int, default=1,
                        help="number of aggregating steps (not implemented yet)")
    parser.add_argument("--val_freq", type=int, default=10,
                        help="frequency of validation per gradient computation")

    parser.add_argument("-B", "--batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, default=0.05,
                        help="base learning rate")
    parser.add_argument("--models_lrs", dest="lrs", type=float, nargs="+", default=None,
                        help="learning rates per model (default: same as base lr if)")

    parser.add_argument("--reg_pow", type=int, choices=(2, 3, 4), default=None,
                        help="power of regularization term (default: no regularization)")
    parser.add_argument("--reg_coeff", type=float, default=1.,
                        help="regularization coefficient")

    parser.add_argument("--save_fig", type=str, default=None,
                        help="save validation plot under this name (default: don't save)")
    parser.add_argument("--log_scale", action="store_true",
                        help="use log scale for validation plot")


    # Parse command line args
    args = parser.parse_args()

    # Handle args specifications
    if args.feature_dim is None:
        args.feature_dim = DATASET_FEATURES[args.dataset]
    if args.lrs is None:
        args.lrs = [args.lr] * args.M
    if args.cuda and torch.cuda.is_available():
        args.device = "cuda:0"  # TODO: assign all available/chosen devices?
    else:
        args.device = "cpu"

    return args


### Dataset ###
class MyDataset(data_utils.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = f"datasets/{dataset}"
        with open(self.dataset, "rb") as f:
            self.X, self.y = load_svmlight_file(f)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        X_tensor = torch.Tensor(self.X[idx].todense()).squeeze(0)
        y_tensor = torch.Tensor([self.y[idx]]).squeeze(0)
        return X_tensor, y_tensor

class DataSampler:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.reset_sampler()

    def reset_sampler(self):
        self.sampler = iter(self.dataloader)

    def sample(self):
        try:
            x, y = next(self.sampler)
        except StopIteration:
            self.reset_sampler()
            x, y = next(self.sampler)
        return x, y


### Optimizer ###
class LocalSGD(torch.optim.SGD):
    def aggregate(self, param_groups=None):
        if param_groups is None:
            param_groups = self.param_groups
        # This function simply averages parameters across all groups/models
        num_params = len(param_groups[0]["params"])
        agg_params = [None] * num_params
        # Find average per parameter
        for param_idx in range(num_params):
            param_list = [param_groups[model_idx]["params"][param_idx].data
                          for model_idx in range(len(param_groups))]
            agg_params[param_idx] = torch.mean(torch.stack(param_list, dim=0), dim=0)

        return agg_params

    def sync(self, agg_params, param_groups=None):
        if param_groups is None:
            param_groups = self.param_groups
        # Synchronize
        num_params = len(param_groups[0]["params"])
        for model_idx in range(len(param_groups)):
            for param_idx in range(num_params):
                param_groups[model_idx]["params"][param_idx] = agg_params[param_idx]

    def aggregate_and_sync(self):
        agg_params = self.aggregate()
        self.sync(agg_params)


### Report Utils ###
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def plot_lines(losses_dict, title="", log_scale=False, show_plot=True, filename=None):
    plt.figure(figsize=(10,5))
    if log_scale:
        title += " (log scale)"
    plt.title(title)
    for label, losses in losses_dict.items():
        if log_scale:
            losses = list(map(log, losses))
        plt.plot(losses, label=label)
    plt.xlabel("t")
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()


### Data Utils ###
def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_seed_worker(seed):
    if seed is None:
        return None
    def seed_worker(worker_id):
        worker_seed = seed % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return seed_worker

def create_generator(seed):
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def create_trainloader(dataset, args):
    return data_utils.DataLoader(dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers, shuffle=True, pin_memory=True,
                                 worker_init_fn=create_seed_worker(args.seed),
                                 generator=create_generator(args.seed))
def create_testloader(dataset, args):
    return data_utils.DataLoader(dataset, batch_size=len(dataset),
                                 num_workers=args.num_workers, shuffle=False, pin_memory=True,
                                 worker_init_fn=create_seed_worker(args.seed),
                                 generator=create_generator(args.seed))


### Aggregation schedule ###
def n_shot_schedule(N, T):
    return {i for i in range(T-1, 0, -T // N)}


### Model ###
def create_model(args):
    return nn.Linear(args.feature_dim, args.output_dim)

def init_model_(model):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            # Use default initialization
            pass
    model.apply(init_weights)

def init_like_(m1, m2):
    with torch.no_grad():
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            p1.copy_(p2)


### Regularization ###
def create_regularizer(p):
    assert p in (2, 3, 4)
    c = (0., 1., 1./2, 1./6, 1./24)
    def regularizer(model):
        params = nn.utils.parameters_to_vector(model.parameters())
        return torch.norm(params).pow(p) * c[p]
    return regularizer


### Training ###
def validate(testloader, model, loss_fn, args):
    with torch.no_grad():
        running_loss = 0.0
        running_acc = 0
        for x, y in testloader:
            x = x.to(device=args.device)
            y = y.to(device=args.device)
            y_pred = model(x).view_as(y)
            loss = loss_fn(y_pred, y)
            acc = torch.sum(y_pred.round() == y).float() / y.size(0)
            running_loss += loss.item()
            running_acc += acc.item()
        mean_loss = running_loss / len(testloader)
        mean_acc = running_acc / len(testloader)
    return mean_loss, mean_acc

def train_models(T, trainloader, testloader, models, optimizer, loss_fn, regularizer, args):
    data_sampler = DataSampler(trainloader)
    stats = defaultdict(list)
    train_losses = AverageMeter("train_loss", ":.4f")
    val_losses = AverageMeter("val_loss", ":.4f")
    progress = ProgressMeter(T, [train_losses, val_losses])
    # Create dummy model for validating aggregated model
    agg_model = create_model(args).to(device=args.device)

    for t in range(T):
        # Validate
        if (t * len(models)) % args.val_freq == 0:
            with torch.no_grad():
                if len(models) > 1:
                    agg_params = optimizer.aggregate()
                    for p1, p2 in zip(agg_model.parameters(), agg_params):
                        p1.copy_(p2)
                    val_loss, _ = validate(testloader, agg_model, loss_fn, args)
                else:
                    val_loss, _ = validate(testloader, models[0], loss_fn, args)
            stats["val_loss"] += [val_loss]
            val_losses.update(val_loss)

        for model in models:
            x, y = data_sampler.sample()
            x = x.to(device=args.device)
            y = y.to(device=args.device)
            y_pred = model(x).view_as(y)
            loss = loss_fn(y_pred, y)
            if regularizer is not None:
                loss = loss + args.reg_coeff * regularizer(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stats["train_loss"] += [loss.item()]
            train_losses.update(loss.item())

        progress.display(t)

    return stats


def main(args):

    # Set seed if given
    if args.seed is not None:
        manual_seed(args.seed)

    # Initialize dataset
    trainset = MyDataset(args.dataset)
    testset = trainset  # test on full dataset

    # Define a group of models (optimized in parallel) and a singled out model (optimized alone)
    one_model = create_model(args).to(device=args.device)
    one_optimizer = torch.optim.SGD(one_model.parameters(), lr=args.lr)
    mult_models = [create_model(args).to(device=args.device) for i in range(args.M)]
    param_groups = [{"params": mult_models[i].parameters(), "lr": args.lrs[i]} for i in range(args.M)]
    mult_optimizer = LocalSGD(param_groups, lr=args.lr)

    # Init models
    init_model_(one_model)
    for model in mult_models:
        init_like_(model, one_model)

    # Loss function
    loss_fn = nn.MSELoss().to(device=args.device)
    regularizer = None
    if args.reg_pow is not None:
        regularizer = create_regularizer(args.reg_pow)

    print("Training single model...")
    stats_one = train_models(args.T * args.M,
                             create_trainloader(trainset, args),
                             create_testloader(testset, args),
                             [one_model], one_optimizer, loss_fn, regularizer, args)
    print("Training multiple models...")
    stats_mult = train_models(args.T,
                              create_trainloader(trainset, args),
                              create_testloader(testset, args),
                              mult_models, mult_optimizer, loss_fn, regularizer, args)

    stats = {
        "mult_train_loss": stats_mult["train_loss"],
        "mult_val_loss": stats_mult["val_loss"],
        "one_train_loss": stats_one["train_loss"],
        "one_val_loss": stats_one["val_loss"],
    }

    train_losses = {k: v for k, v in stats.items() if "train_loss" in k}
    val_losses = {k: v for k, v in stats.items() if "val_loss" in k}
    plot_lines(train_losses, "Training Loss",
               log_scale=args.log_scale, show_plot=args.save_fig is None)
    plot_lines(val_losses, "Validation Loss",
               log_scale=args.log_scale, show_plot=args.save_fig is None, filename=args.save_fig)


if __name__ == "__main__":
    args = parse_args()
    main(args)
