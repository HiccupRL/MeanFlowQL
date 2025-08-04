import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class ReBRACAgent(flax.struct.PyTreeNode):
    """Revisited behavior-regularized actor-critic (ReBRAC) agent.

    ReBRAC is a variant of TD3+BC with layer normalization and separate actor and critic penalization.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the ReBRAC critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select('target_actor')(batch['next_observations'])
        next_actions = next_dist.mode()
        noise = jnp.clip(
            (jax.random.normal(sample_rng, next_actions.shape) * self.config['actor_noise']),
            -self.config['actor_noise_clip'],
            self.config['actor_noise_clip'],
        )
        
        next_actions = jnp.clip(next_actions + noise, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        next_q = next_qs.min(axis=0)

        mse = jnp.square(next_actions - batch['next_actions']).sum(axis=-1)
        next_q = next_q - self.config['alpha_critic'] * mse

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the ReBRAC actor loss."""
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions = dist.mode()

        # Q loss.
        qs = self.network.select('critic')(batch['observations'], actions=actions)
        q = jnp.min(qs, axis=0)

        # BC loss.
        mse = jnp.square(actions - batch['actions']).sum(axis=-1)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        actor_loss = -(lam * q).mean()
        bc_loss = (self.config['alpha_actor'] * mse).mean()

        total_loss = actor_loss + bc_loss

        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev().mean()

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'std': action_std.mean(),
            'mse': mse.mean(),
        }

    @partial(jax.jit, static_argnames=('full_update',))
    def total_loss(self, batch, grad_params, full_update=True, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if full_update:
            # Update the actor.
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            # Skip actor update.
            actor_loss = 0.0

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @partial(jax.jit, static_argnames=('full_update',))
    def update(self, batch, full_update=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if full_update:
            # Update the target networks only when `full_update` is True.
            self.target_update(new_network, 'critic')
            self.target_update(new_network, 'actor')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = dist.mode()
        noise = jnp.clip(
            (jax.random.normal(seed, actions.shape) * self.config['actor_noise'] * temperature),
            -self.config['actor_noise_clip'],
            self.config['actor_noise_clip'],
        )
        actions = jnp.clip(actions + noise, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=False,
            const_std=True,
            final_fc_init_scale=config['actor_fc_scale'],
            encoder=encoders.get('actor'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = params['modules_actor']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='rebrac',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            tanh_squash=True,  # Whether to squash actions with tanh.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            alpha_actor=0.0,  # Actor BC coefficient.
            alpha_critic=0.0,  # Critic BC coefficient.
            actor_freq=2,  # Actor update frequency.
            actor_noise=0.2,  # Actor noise scale.
            actor_noise_clip=0.5,  # Actor noise clipping threshold.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
