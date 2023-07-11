import haiku as hk
import jax
import jax.numpy as jnp
from einops import rearrange, reduce
from typing import Callable, Optional

from utils import Args, Dynamic, ObsModel
from modeling.lib import RecurrentEncoder, GRU_Net
from data.dataset import Batch


class Voe(hk.Module):

    def __init__(self, args: Args, name=None):
        super().__init__(name=name)
        self.args = args

    def __call__(self, obs_seq: jax.Array):
        return RecurrentEncoder(self.args, name='filter')(obs_seq)


class Voe_mlp(hk.Module):

    def __init__(self, args: Args, name=None):
        super().__init__(name=name)
        self.hidden_size = args.hidden_size
        self.output_size = args.dim_state * 2
        self.mlp_size = [self.hidden_size] * args.rnn_depth + [self.output_size]
    
    def __call__(self, obs_state: jax.Array):
        return hk.nets.MLP(self.mlp_size, name='filter')(obs_state)


def create_forward(
    args: Args,
    dynamic: Dynamic,
    prior_logvar: jax.Array,
    obs_model: ObsModel,
    name=None
) -> Callable:

    dynamic = jax.vmap(dynamic)

    def rsample(mu: jax.Array, logvar: jax.Array) -> jax.Array:
        return mu + jnp.exp(0.5 * logvar) * jax.random.normal(hk.next_rng_key(), shape=mu.shape)

    def kl_gaussians(mu1: jax.Array, logvar1: jax.Array, mu2: jax.Array, logvar2: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum(
            jnp.exp(logvar1 - logvar2) + (jnp.square(mu1 - mu2) / jnp.exp(logvar2)) + logvar2 - logvar1 - 1,
            axis=-1
        )

    def forward(batch: Batch, is_training: bool=True):
        print("jit!")
        obs_seq = batch.obs.data # (batch_size, seq_len, obs_dim)

        belief_state = Voe(args, name)(obs_seq)
        mu, logvar = jnp.split(belief_state, 2, axis=-1)

        prior_mu = dynamic(rearrange(mu[:, :-1, :], "b t d -> (b t) d"))
        mu_belief = rearrange(mu[:, 1:, :], "b t d -> (b t) d")
        logvar_belief = rearrange(logvar[:, 1:, :], "b t d -> (b t) d")
        kl_loss = kl_gaussians(mu_belief, logvar_belief, prior_mu, prior_logvar).mean()

        samples = rsample(mu, logvar)
        negative_logprob = - obs_model(samples, obs_seq).mean()
        total_loss = kl_loss + negative_logprob

        metrics = {
            "loss/total": total_loss,
            "loss/kl": kl_loss,
            "loss/n_logp": negative_logprob,
        }
        pred = None

        if not is_training:
            state_seq = batch.state.data # (batch_size, seq_len, state_dim)
            rmse = jnp.sqrt(reduce(jnp.square(state_seq - mu), 'b t d -> d', 'mean'))
            for i, rmse_i in enumerate(rmse):
                metrics[f"rmse/state_{i}"] = rmse_i
            pred = (mu, jnp.exp(0.5 * logvar))

        return pred, total_loss, metrics

    return forward

def create_forward_mlp(
    args: Args,
    dynamic: Dynamic,
    prior_logvar: jax.Array,
    obs_model: ObsModel,
    name=None
) -> Callable:
    dynamic = jax.vmap(dynamic)
    voe = Voe_mlp(args, name)
    mu_prior = jnp.zeros(args.dim_state)
    beta = args.beta

    def rsample(mu: jax.Array, logvar: jax.Array) -> jax.Array:
        return mu + jnp.exp(0.5 * logvar) * jax.random.normal(hk.next_rng_key(), shape=mu.shape)

    def kl_gaussians(mu1: jax.Array, logvar1: jax.Array, mu2: jax.Array, logvar2: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum(
            jnp.exp(logvar1 - logvar2) + (jnp.square(mu1 - mu2) / jnp.exp(logvar2)) + logvar2 - logvar1 - 1,
            axis=-1
        )
    
    def core(state, obs):
        state_ = dynamic(state)
        state_obs = jnp.concatenate([state, state_, obs], axis=-1)
        q_w = voe(state_obs)
        mu, logvar = jnp.split(q_w, 2, axis=-1)
        noise = rsample(mu, logvar)
        state_next = state_ + noise
        state_for_test = state_ + mu
        return state_for_test, (state_next, mu, logvar, state_for_test)

    def forward(batch: Batch, is_training: bool=True):
        obs_seq = batch.obs.data[:, 1:, :]
        state = batch.state.data[:, 0, :] # initial state
        obs_for_scan = rearrange(obs_seq, "b t d -> t b d")
        _, results = hk.scan(core, state, obs_for_scan)
        state_seq, mu_seq, logvar_seq, state_for_test = results

        mu_seq = rearrange(mu_seq, "t b d -> b t d")
        logvar_seq = rearrange(logvar_seq, "t b d -> b t d")
        state_for_test = rearrange(state_for_test, "t b d -> b t d")
        state_seq = rearrange(state_seq, "t b d -> (b t) d")
        obs_seq = rearrange(obs_seq, "b t d -> (b t) d")

        negative_logprob = - obs_model(state_seq, obs_seq).mean()
        kl_loss = kl_gaussians(mu_seq, logvar_seq, mu_prior, prior_logvar).mean()
        total_loss = beta * kl_loss + negative_logprob

        metrics = {
            "loss/total": total_loss,
            "loss/kl": kl_loss,
            "loss/n_logp": negative_logprob,
        }
        pred = None

        if not is_training:
            gt_states = batch.state.data[:, 1:, :] # (batch_size, seq_len-1, state_dim)
            rmse = jnp.sqrt(reduce(jnp.square(gt_states - state_for_test), 'b t d -> d', 'mean'))
            for i, rmse_i in enumerate(rmse):
                metrics[f"rmse/state_{i}"] = rmse_i
            pred = (state_for_test, jnp.exp(0.5 * logvar_seq))

        return pred, total_loss, metrics
    
    return forward

def create_forward_mlp2(
    args: Args,
    dynamic: Dynamic,
    prior_logvar: jax.Array,
    obs_model: ObsModel,
    name=None
) -> Callable:
    dynamic = jax.vmap(dynamic)
    voe = Voe_mlp(args, name)
    mu_prior = jnp.zeros(args.dim_state)
    beta = args.beta

    def rsample(mu: jax.Array, logvar: jax.Array) -> jax.Array:
        return mu + jnp.exp(0.5 * logvar) * jax.random.normal(hk.next_rng_key(), shape=mu.shape)

    def kl_gaussians(mu1: jax.Array, logvar1: jax.Array, mu2: jax.Array, logvar2: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum(
            jnp.exp(logvar1 - logvar2) + (jnp.square(mu1 - mu2) / jnp.exp(logvar2)) + logvar2 - logvar1 - 1,
            axis=-1
        )
    
    def core(state, obs):
        state_ = dynamic(state)
        state_obs = jnp.concatenate([state, state_, obs], axis=-1)
        q_w = voe(state_obs)
        mu, logvar = jnp.split(q_w, 2, axis=-1)
        noise = rsample(mu, logvar)
        state_next = state_ + noise
        state_for_test = state_ + mu
        return state_for_test, (state_next, mu, logvar, state_for_test)

    def forward(batch: Batch, is_training: bool=True):
        obs_seq = batch.obs.data
        obs_hidden = GRU_Net(
            hidden_size=args.hidden_size,
            rnn_depth=2,
            name="obs_gru"
        )(obs_seq)
        state = batch.state.data[:, 0, :] # initial state
        obs_for_scan = rearrange(obs_hidden[:, 1:], "b t d -> t b d")
        _, results = hk.scan(core, state, obs_for_scan)
        state_seq, mu_seq, logvar_seq, state_for_test = results

        mu_seq = rearrange(mu_seq, "t b d -> b t d")
        logvar_seq = rearrange(logvar_seq, "t b d -> b t d")
        state_for_test = rearrange(state_for_test, "t b d -> b t d")
        state_seq = rearrange(state_seq, "t b d -> (b t) d")
        obs_seq = rearrange(obs_seq[:, 1:], "b t d -> (b t) d")
        negative_logprob = - obs_model(state_seq, obs_seq).mean()
        kl_loss = kl_gaussians(mu_seq, logvar_seq, mu_prior, prior_logvar).mean()
        total_loss = beta * kl_loss + negative_logprob

        metrics = {
            "loss/total": total_loss,
            "loss/kl": kl_loss,
            "loss/n_logp": negative_logprob,
        }
        pred = None

        if not is_training:
            gt_states = batch.state.data[:, 1:, :] # (batch_size, seq_len-1, state_dim)
            rmse = jnp.sqrt(reduce(jnp.square(gt_states - state_for_test), 'b t d -> d', 'mean'))
            for i, rmse_i in enumerate(rmse):
                metrics[f"rmse/state_{i}"] = rmse_i
            pred = (state_for_test, jnp.exp(0.5 * logvar_seq))

        return pred, total_loss, metrics
    
    return forward