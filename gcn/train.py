import sys 
sys.path.append('./python')

import time
import argparse
import numpy as np


from utils import load_data, masked_accuracy
from model import GCN
from needle import optim
import needle as ndl


def make_mask(indices, size, device):
    mask = np.zeros(size, dtype=np.float32)
    mask[indices] = 1
    return ndl.Tensor(mask, device=device, dtype="float32")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda

np.random.seed(args.seed)
if args.cuda:
    # torch.cuda.manual_seed(args.seed)
    # TODO: set needle cuda seed
    pass

# Load data
device = ndl.cuda() if args.cuda else ndl.cpu()
adj, features, labels, idx_train, idx_val, idx_test = load_data("data/cora/")
adj = ndl.autograd.SparseTensor.make_from_numpy(adj.row, adj.col, adj.data.astype(np.float32), shape=adj.shape, device=device)
features = ndl.Tensor(features, device=device, dtype="float32", requires_grad=False)
labels = ndl.Tensor(labels, device=device, dtype="float32", requires_grad=False)
train_mask = make_mask(idx_train, labels.shape[0], device)
val_mask = make_mask(idx_val, labels.shape[0], device)
test_mask = make_mask(idx_test, labels.shape[0], device)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=7,
            dropout=args.dropout)
loss_module = ndl.nn.MaskedSoftmaxLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model = model.cuda()
    loss_module = loss_module.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.reset_grad()
    output = model(features, adj)
    loss_train, loss_val, loss_test = loss_module(output, labels, (train_mask, val_mask, test_mask))
    acc_train, acc_val, acc_test = masked_accuracy(output, labels, (train_mask, val_mask, test_mask))
    breakpoint()
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = loss_module(output, labels, (train_mask, val_mask, test_mask))[2]
    acc_test = masked_accuracy(output, labels, (train_mask, val_mask, test_mask))[2]
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()