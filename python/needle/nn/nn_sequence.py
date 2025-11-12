"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ex = ops.exp(x)
        return ex / (1 + ex)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_flag = bias
        assert nonlinearity in ("tanh", "relu")
        self.nonlinearity = nonlinearity
        bound = 1.0 / math.sqrt(hidden_size)
        self.W_ih = Parameter(
            init.rand(input_size, hidden_size,
                      low=-bound, high=bound,
                      device=device, dtype=dtype)
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, hidden_size,
                      low=-bound, high=bound,
                      device=device, dtype=dtype)
        )
        if bias:
            self.bias_ih = Parameter(
                init.rand(hidden_size,
                          low=-bound, high=bound,
                          device=device, dtype=dtype)
            )
            self.bias_hh = Parameter(
                init.rand(hidden_size,
                          low=-bound, high=bound,
                          device=device, dtype=dtype)
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size,
                           device=X.device, dtype=X.dtype)

        # X @ W_ih + bias_ih
        h_in = X @ self.W_ih
        if self.bias_ih is not None:
            b_ih = ops.reshape(self.bias_ih, (1, self.hidden_size))
            b_ih = ops.broadcast_to(b_ih, h_in.shape)
            h_in = h_in + b_ih

        # h @ W_hh + bias_hh
        h_rec = h @ self.W_hh
        if self.bias_hh is not None:
            b_hh = ops.reshape(self.bias_hh, (1, self.hidden_size))
            b_hh = ops.broadcast_to(b_hh, h_rec.shape)
            h_rec = h_rec + b_hh

        pre_act = h_in + h_rec

        if self.nonlinearity == "tanh":
            h_next = ops.tanh(pre_act)
        else:
            h_next = ops.relu(pre_act)

        return h_next
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias_flag = bias

        self.rnn_cells = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            cell = RNNCell(in_size, hidden_size,
                           bias=bias,
                           nonlinearity=nonlinearity,
                           device=device, dtype=dtype)
            self.rnn_cells.append(cell)
            setattr(self, f"rnn_cell_{layer}", cell)
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size,
                            device=X.device, dtype=X.dtype)
        h_layers_tuple = ops.split(h0, axis=0)
        h_layers = []
        for hl in h_layers_tuple:
            h_layers.append(ops.reshape(hl, (bs, self.hidden_size)))

        # split sequence along time: each x_t_raw: (1, bs, input_size)
        x_time_steps = ops.split(X, axis=0)

        outputs_last_layer = []

        for t in range(seq_len):
            x_t_raw = x_time_steps[t]
            x_t = ops.reshape(x_t_raw, (bs, self.input_size))

            for layer_idx, cell in enumerate(self.rnn_cells):
                h_prev = h_layers[layer_idx]
                h_new = cell(x_t, h_prev)
                h_layers[layer_idx] = h_new
                x_t = h_new  # pass to next layer

            outputs_last_layer.append(x_t)

        # stack outputs along time: (seq_len, bs, hidden_size)
        output = ops.stack(outputs_last_layer, axis=0)

        # final hidden states for all layers: list of (bs, hidden) -> (num_layers, bs, hidden)
        h_n = ops.stack(h_layers, axis=0)

        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_flag = bias
        self.device = device
        self.dtype = dtype

        bound = 1.0 / math.sqrt(hidden_size)

        # (input_size, 4*hidden_size)
        self.W_ih = Parameter(
            init.rand(input_size, 4 * hidden_size,
                      low=-bound, high=bound,
                      device=device, dtype=dtype)
        )
        # (hidden_size, 4*hidden_size)
        self.W_hh = Parameter(
            init.rand(hidden_size, 4 * hidden_size,
                      low=-bound, high=bound,
                      device=device, dtype=dtype)
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(4 * hidden_size,
                          low=-bound, high=bound,
                          device=device, dtype=dtype)
            )
            self.bias_hh = Parameter(
                init.rand(4 * hidden_size,
                          low=-bound, high=bound,
                          device=device, dtype=dtype)
            )
        else:
            self.bias_ih = None
            self.bias_hh = None

        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]

        if h is None:
            h0 = init.zeros(bs, self.hidden_size,
                            device=X.device, dtype=X.dtype)
            c0 = init.zeros(bs, self.hidden_size,
                            device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h

        # Linear part for all gates at once: (bs, 4H)
        gates = X @ self.W_ih + h0 @ self.W_hh

        if self.bias_ih is not None:
            b_ih = ops.reshape(self.bias_ih, (1, 4 * self.hidden_size))
            b_ih = ops.broadcast_to(b_ih, gates.shape)
            gates = gates + b_ih

        if self.bias_hh is not None:
            b_hh = ops.reshape(self.bias_hh, (1, 4 * self.hidden_size))
            b_hh = ops.broadcast_to(b_hh, gates.shape)
            gates = gates + b_hh

        # gates: (bs, 4H) -> (bs, 4, H) -> split along axis=1
        gates_3d = ops.reshape(gates, (bs, 4, self.hidden_size))
        i3, f3, g3, o3 = ops.split(gates_3d, axis=1)  # each (bs,1,H)

        i = ops.reshape(i3, (bs, self.hidden_size))
        f = ops.reshape(f3, (bs, self.hidden_size))
        g = ops.reshape(g3, (bs, self.hidden_size))
        o = ops.reshape(o3, (bs, self.hidden_size))

        i = self.sigmoid(i)
        f = self.sigmoid(f)
        o = self.sigmoid(o)
        g = ops.tanh(g)

        c_next = f * c0 + i * g
        h_next = o * ops.tanh(c_next)

        return h_next, c_next
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias_flag = bias
        self.device = device
        self.dtype = dtype

        self.lstm_cells = []
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            cell = LSTMCell(in_size, hidden_size,
                            bias=bias,
                            device=device, dtype=dtype)
            self.lstm_cells.append(cell)
            setattr(self, f"lstm_cell_{layer}", cell)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        if h is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size,
                            device=X.device, dtype=X.dtype)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size,
                            device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h

        # Split initial states per layer: (1,bs,H) -> (bs,H)
        h_layers_raw = ops.split(h0, axis=0)
        c_layers_raw = ops.split(c0, axis=0)
        h_layers = [ops.reshape(h_l, (bs, self.hidden_size))
                    for h_l in h_layers_raw]
        c_layers = [ops.reshape(c_l, (bs, self.hidden_size))
                    for c_l in c_layers_raw]
        x_time_steps = ops.split(X, axis=0)
        outputs_last_layer = []

        for t in range(seq_len):
            x_t_raw = x_time_steps[t]          # (1,bs,input_size)
            x_t = ops.reshape(x_t_raw, (bs, self.input_size))

            for layer_idx, cell in enumerate(self.lstm_cells):
                h_prev = h_layers[layer_idx]
                c_prev = c_layers[layer_idx]
                h_new, c_new = cell(x_t, (h_prev, c_prev))
                h_layers[layer_idx] = h_new
                c_layers[layer_idx] = c_new
                x_t = h_new  # pass hidden to next layer

            outputs_last_layer.append(x_t)

        output = ops.stack(outputs_last_layer, axis=0)
        h_n = ops.stack(h_layers, axis=0)
        c_n = ops.stack(c_layers, axis=0)
        return output, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x_nd = x.realize_cached_data()         # NDArray
        x_np = x_nd.numpy().astype("int64")    # (seq_len, bs)
        flat_ids = x_np.reshape(-1)            # (N,)
        W_nd = self.weight.realize_cached_data()
        dev = W_nd.device
        one_hot_rows = [dev.one_hot(self.num_embeddings, int(idx)).numpy()
                        for idx in flat_ids]
        one_hot_np = np.stack(one_hot_rows, axis=0).astype("float32")  # (N, num_embeddings)
        one_hot = Tensor(one_hot_np, device=dev)
        emb_flat = one_hot @ self.weight
        return emb_flat.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION