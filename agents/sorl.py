# sorl.py
import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Value, ShortcutModel


class SORLAgent(flax.struct.PyTreeNode):
    """Scalable Offline Reinforcement Learning (SORL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    step: jnp.ndarray = 0

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        switch_rng, idx_rng = jax.random.split(rng, 2)
        fns = [
            (
                lambda k_val: (
                    lambda args: self.sample_k_step_actions_for_critic(*args, k=k_val)
                )
            )(k)
            for k in self.config["k_values"]
        ]
        idx = jax.random.randint(switch_rng, shape=(), minval=0, maxval=len(fns))
        next_actions = jax.lax.switch(idx, fns, (batch, grad_params, idx_rng))
        timesteps = jnp.ones((batch["next_observations"].shape[0], 1)) * (2**idx)

        next_qs = self.network.select("target_critic")(
            batch["next_observations"], actions=next_actions, timesteps=timesteps
        )
        if self.config["q_agg"] == "min":
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q

        q = self.network.select("critic")(
            batch["observations"],
            actions=batch["actions"],
            timesteps=timesteps,
            params=grad_params,
        )
        td_errors = q - target_q
        critic_loss = jnp.square(td_errors).mean()

        td_error_mean = jnp.mean(td_errors)
        td_error_std = jnp.std(td_errors)

        target_mean = jnp.mean(target_q)
        target_var = jnp.mean(jnp.square(target_q - target_mean))
        mse = jnp.mean(jnp.square(td_errors))
        explained_variance = 1.0 - (mse / (jnp.maximum(target_var, 1e-8)))

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "explained_variance": explained_variance,
            "td_error_mean": td_error_mean,
            "td_error_std": td_error_std,
        }

    def sample_k_step_actions_for_critic(self, batch, grad_params, rng, k=1):
        """Sample k-step actions using next_observations from batch."""
        observations = batch["next_observations"]
        actor_actions = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                self.config["action_dim"],
            ),
        )
        delta_t = 1.0 / k
        for ti in range(k):
            t = ti / k  # from x_0 (noise) to x_1
            if len(observations.shape) > 1:
                t_vector = jnp.full((observations.shape[0], 1), t)
                dt_flow = jnp.log2(k).astype(jnp.int32)
                dt_base = (
                    jnp.ones((observations.shape[0], 1), dtype=jnp.int32) * dt_flow
                )
            else:
                t_vector = jnp.full((1,), t)
                dt_flow = jnp.log2(k).astype(jnp.int32)
                dt_base = jnp.ones((1,), dtype=jnp.int32) * dt_flow
            v = self.network.select("actor")(
                observations, actor_actions, t_vector, dt_base, params=grad_params
            )
            actor_actions = actor_actions + v * delta_t  # Euler sampling
        actor_actions = jnp.clip(actor_actions, -1, 1)
        return actor_actions

    def sample_k_step_actions_for_q(self, batch, grad_params, rng, k=1):
        """Sample k-step actions using observations from batch."""
        observations = batch["observations"]
        actor_actions = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config["ob_dims"])],
                self.config["action_dim"],
            ),
        )
        delta_t = 1.0 / k
        for ti in range(k):
            t = ti / k
            if len(observations.shape) > 1:
                t_vector = jnp.full((observations.shape[0], 1), t)
                dt_flow = jnp.log2(k).astype(jnp.int32)
                dt_base = (
                    jnp.ones((observations.shape[0], 1), dtype=jnp.int32) * dt_flow
                )
            else:
                t_vector = jnp.full((1,), t)
                dt_flow = jnp.log2(k).astype(jnp.int32)
                dt_base = jnp.ones((1,), dtype=jnp.int32) * dt_flow
            v = self.network.select("actor")(
                observations, actor_actions, t_vector, dt_base, params=grad_params
            )
            actor_actions = actor_actions + v * delta_t
        actor_actions = jnp.clip(actor_actions, -1, 1)
        return actor_actions

    def get_consistency_loss(self, batch, grad_params, rng):
        t_rng, x_rng = jax.random.split(rng, 2)
        batch_size, _ = batch["actions"].shape
        log2_sections = np.log2(self.config["denoise_timesteps"]).astype(np.int32)
        dt_base = jnp.repeat(
            log2_sections - 1 - jnp.arange(log2_sections), batch_size // log2_sections
        )
        dt_base = jnp.concatenate(
            [
                dt_base,
                jnp.zeros(
                    batch_size - dt_base.shape[0],
                ),
            ]
        )
        dt = 1 / (2 ** (dt_base))
        dt_base_bootstrap = dt_base + 1
        dt_bootstrap = dt / 2

        # sample t
        dt_sections = jnp.power(2, dt_base)
        t = jax.random.randint(
            t_rng, (batch_size,), minval=0, maxval=dt_sections
        ).astype(jnp.float32)
        t = t / dt_sections  # between 0 and 1
        t_full = t[:, None]

        # generate bootstrap targets
        x_1 = batch["actions"]
        x_0 = jax.random.normal(x_rng, x_1.shape)
        x_t = (1 - (1 - self.config["t_epsilon"]) * t_full) * x_0 + t_full * x_1
        bootstrap_pred = self.network.select("actor")(
            batch["observations"],
            x_t,
            t[:, None],
            dt_base[:, None],
            params=grad_params,
        )
        v_b1 = self.network.select("target_actor")(
            batch["observations"], x_t, t[:, None], dt_base_bootstrap[:, None]
        )
        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None] * v_b1

        v_b2 = self.network.select("target_actor")(
            batch["observations"], x_t2, t2[:, None], dt_base_bootstrap[:, None]
        )
        v_target = (v_b1 + v_b2) / 2
        consistency_loss = jnp.mean((bootstrap_pred - v_target) ** 2)
        return consistency_loss

    def get_fm_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch["actions"].shape
        x_rng, t_rng = jax.random.split(rng, 2)
        t = jax.random.randint(
            t_rng, (batch_size,), minval=0, maxval=self.config["denoise_timesteps"]
        ).astype(np.float32)
        t /= self.config["denoise_timesteps"]
        t_full = t[:, None]

        # sample flow pairs x_t, v_t
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch["actions"]
        x_t = (1 - (1 - self.config["t_epsilon"]) * t_full) * x_0 + t_full * x_1
        v_t = x_1 - (1 - self.config["t_epsilon"]) * x_0
        dt_flow = np.log2(self.config["denoise_timesteps"]).astype(np.int32)
        dt_base = jnp.ones(batch_size, dtype=np.int32) * dt_flow

        fm_pred = self.network.select("actor")(
            batch["observations"],
            x_t,
            t[:, None],
            dt_base[:, None],
            params=grad_params,
        )
        bc_flow_loss = jnp.mean((fm_pred - v_t) ** 2)
        return bc_flow_loss

    def get_q_loss(self, batch, grad_params, rng):
        switch_rng, idx_rng = jax.random.split(rng, 2)
        fns = [
            (
                lambda k_val: (
                    lambda args: self.sample_k_step_actions_for_q(*args, k=k_val)
                )
            )(k)
            for k in self.config["k_values"]
        ]
        idx = jax.random.randint(switch_rng, shape=(), minval=0, maxval=len(fns))
        actor_actions = jax.lax.switch(idx, fns, (batch, grad_params, idx_rng))
        timesteps = jnp.ones((batch["observations"].shape[0], 1)) * (2**idx)

        qs = self.network.select("critic")(
            batch["observations"], actions=actor_actions, timesteps=timesteps
        )
        q = jnp.mean(qs, axis=0)
        q_loss = -q.mean()
        if self.config["normalize_q_loss"]:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        return (q_loss, actor_actions, q)

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        consistency_rng, fm_rng, q_rng = jax.random.split(rng, 3)
        consistency_loss = self.get_consistency_loss(
            batch=batch, grad_params=grad_params, rng=consistency_rng
        )
        fm_loss = self.get_fm_loss(batch=batch, grad_params=grad_params, rng=fm_rng)
        q_loss, actor_actions, q = self.get_q_loss(
            batch=batch, grad_params=grad_params, rng=q_rng
        )

        actor_loss = (
            (self.config["bc_flow_loss_coef"] * fm_loss)
            + (self.config["alpha"] * consistency_loss)
            + (self.config["q_loss_coef"] * q_loss)
        )

        mse = jnp.mean((actor_actions - batch["actions"]) ** 2)

        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_flow_loss": fm_loss,
            "consistency_loss": consistency_loss,
            "mse": mse,
            "alpha_consistency_loss": self.config["alpha"] * consistency_loss,
            "q_loss": q_loss,
            "scaled_q_loss": self.config["q_loss_coef"] * q_loss,
            "q": q.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v
        info["actor/alpha"] = self.config["alpha"]

        loss = critic_loss + actor_loss
        return loss, info

    def critic_target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    def actor_target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        # critic update
        def critic_loss_fn(grad_params):
            loss, info = self.critic_loss(batch, grad_params, rng=rng)
            return loss, info

        new_network, critic_info = self.network.apply_loss_fn(loss_fn=critic_loss_fn)
        self.critic_target_update(new_network, "critic")

        # actor update
        actor_info = {}
        for _ in range(self.config["actor_updates_per_batch"]):
            rng, subrng = jax.random.split(rng)

            def actor_loss_fn(grad_params):
                loss, info = self.actor_loss(batch, grad_params, rng=subrng)
                return loss, info

            new_network, a_info = new_network.apply_loss_fn(loss_fn=actor_loss_fn)
            actor_info = a_info
        self.actor_target_update(new_network, "actor")

        info = {**critic_info, **actor_info}
        return self.replace(network=new_network, rng=new_rng, step=self.step + 1), info

    @partial(jax.jit, static_argnames=("inference_timesteps", "best_of_d", "n"))
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature: float = 1.0,  # keep for compatibility with evaluation.py
        inference_timesteps: int = 1,
        best_of_d: bool = True,
        n: int = 1,
    ):
        """Sample actions using either best-of-N with fixed steps or best-of-4 with varying steps.

        When config['best_of_d'] is True: Generate 4 candidates with k ∈ {1, 2, 4, 8} steps.
        When config['best_of_d'] is False: Generate N candidates with config['inference_timesteps'] steps.

        In both cases, select the action that maximizes the Q-value.
        """
        added_batch_dim = False
        if observations.ndim == len(self.config["ob_dims"]):  # single sample
            observations = observations[None, ...]
            added_batch_dim = True
        batch_size = observations.shape[0]

        denoise_steps = inference_timesteps
        N = n
        delta_t = 1.0 / denoise_steps
        dt_flow = np.log2(denoise_steps).astype(jnp.int32)

        def _one_candidate(rng_key):
            x = jax.random.normal(
                rng_key,
                observations.shape[:-1] + (self.config["action_dim"],),
            )
            for ti in range(denoise_steps):
                t = ti / denoise_steps
                t_vec = jnp.full(observations.shape[:-1] + (1,), t)
                dt_base = jnp.ones_like(t_vec, dtype=jnp.int32) * dt_flow
                v = self.network.select("actor")(observations, x, t_vec, dt_base)
                x = x + v * delta_t
            return jnp.clip(x, -1, 1)

        keys = jax.random.split(self.rng if seed is None else seed, N)
        candidate_actions = jax.vmap(_one_candidate)(keys)

        if N == 1:
            best_actions = candidate_actions[0]

            if added_batch_dim:
                best_actions = best_actions[0]
            return best_actions

        # evaluate with the critic
        cand_shape = candidate_actions.shape
        flat_actions = candidate_actions.reshape((-1,) + cand_shape[2:])
        flat_obs = jnp.repeat(observations, N, axis=0)
        t_shape = flat_obs.shape[:-1] + (1,)
        flat_t = jnp.full(t_shape, denoise_steps)

        # common evaluation and selection code
        qs = self.network.select("critic")(
            flat_obs, actions=flat_actions, timesteps=flat_t
        )
        if self.config["q_agg"] == "mean":
            q_mean = qs.mean(axis=0)
        elif self.config["q_agg"] == "min":
            q_mean = qs.min(axis=0)
        else:
            raise ValueError(f"Invalid q_agg: {self.config['q_agg']}")

        q_mean = q_mean.reshape((N, batch_size, -1)).mean(axis=-1).T

        # select best candidate per batch element
        best_idx = jnp.argmax(q_mean, axis=1)

        # select best actions
        def _pick(cands, idx):
            return cands[idx, ...]

        best_actions = jax.vmap(_pick, in_axes=(1, 0))(candidate_actions, best_idx)

        if added_batch_dim:
            best_actions = best_actions[0]

        return best_actions

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

        ex_d = ex_actions[..., :1]
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # define encoders
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor"] = encoder_module()

        # define networks
        critic_def = Value(
            hidden_dims=(config["value_layer_size"],) * config["value_num_layers"],
            layer_norm=config["layer_norm"],
            num_ensembles=2,
            encoder=encoders.get("critic"),
        )

        actor_def = ShortcutModel(
            hidden_dims=(config["actor_layer_size"],) * config["actor_num_layers"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor"),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions, ex_times)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_actions, ex_times),
            ),
            actor=(actor_def, (ex_observations, ex_actions, ex_times, ex_d)),
            target_actor=(
                copy.deepcopy(actor_def),
                (ex_observations, ex_actions, ex_times, ex_d),
            ),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)

        optimizer_mapping = {
            "adam": optax.adam,
            "adamw": optax.adamw,
        }
        transformations = []
        if config["grad_clip_norm"] > 0:
            transformations.append(optax.clip_by_global_norm(config["grad_clip_norm"]))
        optimizer_fn = optimizer_mapping[config["optimizer"]]

        transformations.append(
            optimizer_fn(
                learning_rate=config["lr"],
                b1=config["optimizer_beta1"],
                b2=config["optimizer_beta2"],
                **(
                    {"weight_decay": config["optimizer_weight_decay"]}
                    if config["optimizer"] == "adamw"
                    else {}
                ),
            )
        )
        network_tx = optax.chain(*transformations)
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(
            network_def,
            network_params,
            tx=network_tx,
            grad_clip_norm=config["grad_clip_norm"],
        )

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]
        params["modules_target_actor"] = params["modules_actor"]

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim

        # generate list of k values: powers of 2 starting from 1 up to max_k_backprop
        k_values = []
        k = 1
        while k <= config["max_k_backprop"]:
            k_values.append(k)
            k *= 2
        config["k_values"] = k_values

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="sorl",  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(
                list
            ),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(
                int
            ),  # Action dimension (will be set automatically).
            lr=1e-4,  # Learning rate.
            grad_clip_norm=1.0,  # Gradient clipping norm (0.0 means no clipping)
            optimizer="adam",
            optimizer_beta1=0.9,  # Beta1 for the optimizer.
            optimizer_beta2=0.999,  # Beta2 for the optimizer.
            optimizer_weight_decay=0.0,  # Weight decay for the optimizer.
            batch_size=256,  # Batch size.
            actor_layer_size=512,
            actor_num_layers=4,
            value_layer_size=512,
            value_num_layers=4,
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            bc_flow_loss_coef=10.0,
            tau=0.005,  # 0.005,  # Target network update rate.
            q_agg="mean",  # Aggregation method for target Q values.
            alpha=10.0,  # SC coefficient
            q_loss_coef=100.0,  # Q loss coefficient
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            t_epsilon=1e-5,
            actor_updates_per_batch=1,
            denoise_timesteps=8,
            max_k_backprop=8,
            best_of_n_sweep=(1,),
        )
    )
    return config
