import haiku as hk
import jax
import jax.numpy as jnp
from typing import Callable

from AB3DMOT_libs.utils_voe import Args


def get_activation(activation):
    if activation == "relu":
        return jax.nn.relu
    elif activation == "sigmoid":
        return jax.nn.sigmoid
    elif activation is None:
        return lambda x: x
    else:
        raise ValueError(f"Unknown activation: {activation}")


class Projection(hk.Module):

    def __init__(self, hidden_size: int, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        # self.activation = get_activation("relu")

    def __call__(self, x):
        x = hk.Linear(self.hidden_size)(x)
        return jax.nn.relu(x)


class RecurrentEncoder(hk.Module):

    def __init__(self, args: "Args", name=None):
        super().__init__(name=name)
        self.hidden_size = args.hidden_size
        self.rnn_depth = args.rnn_depth
        self.output_size = args.dim_state * 2
        self.rnn_activation = args.rnn_activation
    
    def make_rnn(self):
        model_list = []
        # model_list.append(Projection(self.hidden_size))
        for i in range(self.rnn_depth):
            model_list.append(hk.LSTM(self.hidden_size))
            if i < self.rnn_depth - 1 and self.rnn_activation:
                model_list.append(jax.nn.relu)
        # model_list.append(hk.Linear(self.output_size))
        model = hk.DeepRNN(
            model_list,
            name="rnn"
        )
        return model

    def __call__(self, x: jax.Array):
        batch_size, *_ = x.shape
        x = Projection(self.hidden_size, name="projection")(x)
        rnn = self.make_rnn()
        initial_state = rnn.initial_state(batch_size)
        x, _ = hk.dynamic_unroll(rnn, x, initial_state, time_major=False)
        # x, _ = hk.static_unroll(rnn, x, initial_state, time_major=False)
        x = hk.Linear(self.output_size, "linear")(x)
        return x


class GRU_Net(hk.Module):

    def __init__(self, hidden_size: int, rnn_depth: int, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.rnn_depth = rnn_depth
    
    def make_rnn(self):
        model_list = []
        for i in range(self.rnn_depth):
            model_list.append(hk.GRU(self.hidden_size))
            if i < self.rnn_depth - 1:
                model_list.append(jax.nn.relu)
        model = hk.DeepRNN(model_list, name="rnn")
        return model
    
    def __call__(self, x: jax.Array):
        batch_size, *_ = x.shape
        rnn = self.make_rnn()
        initial_state = rnn.initial_state(batch_size)
        x, _ = hk.dynamic_unroll(rnn, x, initial_state, time_major=False)
        return x


class MlpEncoder(hk.Module):
    def __init__(self, args: "Args", name=None):
        super().__init__(name=name)
        self.hidden_size = args.hidden_size
        self.output_size = args.dim_state * 2
        self.mlp_size = [self.hidden_size] * args.rnn_depth + [self.output_size]

    def __call__(self, x):
        return hk.nets.MLP(self.mlp_size)(x)


class Decoder(hk.Module):

    def __init__(self, args: "Args", name=None):
        super().__init__(name=name)
        hidden_size = args.hidden_size
        decoder_depth = args.decoder_depth
        output_size = args.dim_obs
        self.mlp_sizes = [hidden_size] * (decoder_depth - 1) + [output_size]

    def __call__(self, x):
        return hk.nets.MLP(self.mlp_sizes)(x)

# def forward(x):
#     return Projection(256)(x)

# if __name__ == '__main__':
#     # args = Args()
#     transformed = hk.transform(forward)
#     params = transformed.init(jax.random.PRNGKey(42), jnp.ones((1, 256)))
#     y = transformed.apply(params, jax.random.PRNGKey(42), jnp.ones((1, 256)))
#     print(y.shape)