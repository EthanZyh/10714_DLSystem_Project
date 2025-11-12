import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBN(ndl.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,device=None,dtype="float32"):
        super().__init__()
        if isinstance(kernel_size,tuple): kernel_size = kernel_size[0]
        if isinstance(stride,tuple): stride = stride[0]
        self.conv = ndl.nn.Conv(in_channels,out_channels,kernel_size,stride=stride,device=device,dtype=dtype)
        self.bn = ndl.nn.BatchNorm1d(out_channels,device=device,dtype=dtype)

    def forward(self,x):
        # x: (N,C,H,W)
        x = self.conv(x) # (N,C,H,W)
        N,C,H,W = x.shape
        x = ndl.ops.transpose(x,(1,2)) # (N,H,C,W)
        x = ndl.ops.transpose(x,(2,3)) # (N,H,W,C)
        x = ndl.ops.reshape(x,(N*H*W,C))
        x = self.bn(x) 
        x = ndl.ops.reshape(x,(N,H,W,C))
        x = ndl.ops.transpose(x,(2,3)) # (N,H,C,W)
        x = ndl.ops.transpose(x,(1,2)) # (N,C,H,W)
        x = ndl.ops.relu(x)
        return x

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.convbn1 = ConvBN(3,16,7,4,device=device,dtype=dtype)
        self.convbn2 = ConvBN(16,32,3,2,device=device,dtype=dtype)
        self.convbn3 = ConvBN(32,32,3,1,device=device,dtype=dtype)
        self.convbn4 = ConvBN(32,32,3,1,device=device,dtype=dtype)
        self.convbn5 = ConvBN(32,64,3,2,device=device,dtype=dtype)
        self.convbn6 = ConvBN(64,128,3,2,device=device,dtype=dtype)
        self.convbn7 = ConvBN(128,128,3,1,device=device,dtype=dtype)
        self.convbn8 = ConvBN(128,128,3,1,device=device,dtype=dtype)
        self.flatten = ndl.nn.Flatten()
        self.fc1 = ndl.nn.Linear(128,128,device=device,dtype=dtype)
        self.fc2 = ndl.nn.Linear(128,10,device=device,dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.convbn1(x)
        x = self.convbn2(x)
        res = x
        x = self.convbn3(x)
        x = self.convbn4(x)
        x = x + res
        x = self.convbn5(x)
        x = self.convbn6(x)
        res = x
        x = self.convbn7(x)
        x = self.convbn8(x)
        x = x + res
        x = self.flatten(x)
        x = ndl.ops.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_model_type = seq_model
        self.seq_len = seq_len

        # Embedding: vocab size = output_size
        self.embedding = nn.Embedding(output_size, embedding_size,
                                   device=device, dtype=dtype)

        # Sequence model
        if seq_model == "rnn":
            self.seq_model = nn.RNN(embedding_size, hidden_size,
                                    num_layers=num_layers,
                                    nonlinearity="tanh",
                                    device=device, dtype=dtype)
        elif seq_model == "lstm":
            self.seq_model = nn.LSTM(embedding_size, hidden_size,
                                     num_layers=num_layers,
                                     device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown seq_model: {seq_model}")

        # Linear head: hidden -> vocab logits
        self.fc = nn.Linear(hidden_size, output_size,
                            device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        emb = self.embedding(x)

        # Sequence model
        seq_out, h_new = self.seq_model(emb, h)   # (seq_len, bs, hidden_size)

        seq_len, bs, hidden = seq_out.shape
        flat = seq_out.reshape((seq_len * bs, hidden))
        logits = self.fc(flat)                    # (seq_len*bs, output_size)
        return logits, h_new
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
