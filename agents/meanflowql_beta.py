import copy
from typing import Any

import flax
import jax
import jax.nn as nn
import jax.numpy as jnp
import ml_collections
import optax
from absl import flags
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
from utils.dit_jax import MFDiT, MFDiT_SIM_e


def inv_softsign(x, scale_norm_factor=0.5):
    """Inverse softsign function: inv_softsign(x) = x / (1 - |x|)"""
    # Ensure that the input is within the valid range (-1, 1).
    x_clipped = jnp.clip(x, -1 + 1e-7, 1 - 1e-7)
    # Calculate the inverse softsign
    result = x_clipped / (1 - jnp.abs(x_clipped))
    # Apply scaling factor
    return result * (1 - scale_norm_factor)


class MeanFlowQL_Agent_BETA(flax.struct.PyTreeNode):
    """Mean Flow Q-learning (MeanFQL) agent.
    Hiccup: this is code based on meanflow_dit and use gn = e-fn(e,r,t), set r=0 as the base, and use discrete time schedule. 
    using consistency loss, meanflow_loss and bound_loss, 
    # this version enabled different learning rate for different.
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def sample_noise(self, rng, shape):
        # Choose your initial distributions. 
        noise_type = self.config.get('noise_type', 'gaussian')
        
        if noise_type == 'gaussian':
            # Scale the noise to sigma. 
            sigma = self.config.get("sigma", 1.0)
            if sigma <= 0:
                raise ValueError(f"sigma must be positive, got {sigma}")
            return jax.random.normal(rng, shape) * sigma
        elif noise_type == 'uniform':
            return jax.random.uniform(rng, shape, minval=-1.0, maxval=1.0)
        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}. Supported types: 'gaussian', 'uniform'")
    
    def critic_loss(self, batch, grad_params, rng):
        """Critic Function Loss"""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        
        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        elif self.config['q_agg'] == 'max':
            next_q = next_qs.max(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean() 

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }
    

    def meanflow_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, t_rng, r_rng, noise_rng = jax.random.split(rng, 4)
        # Discrete time schedule with optional t=1 probability boost
        time_steps = self.config.get('time_steps', 10)
        t_one_prob = self.config.get('t_one_prob', 0.0)
        
        if time_steps <= 1000:
            time_values = jnp.linspace(1/time_steps, 1.0, time_steps)
            indices = jax.random.randint(t_rng, (batch_size,), 0, time_steps)
            t_normal = time_values[indices]
            
            # Apply t=1 probability boost only when t_one_prob > 0
            if t_one_prob > 0.0:
                t_one_mask = jax.random.uniform(t_rng, (batch_size,)) < t_one_prob
                t = jnp.where(t_one_mask, 1.0, t_normal).reshape(-1, 1)
            else:
                t = t_normal.reshape(-1, 1)
        else:
            t_uniform = jax.random.uniform(t_rng, (batch_size,))
            
            # Apply t=1 probability boost only when t_one_prob > 0
            if t_one_prob > 0.0:
                t_one_mask = jax.random.uniform(t_rng, (batch_size,)) < t_one_prob
                t = jnp.where(t_one_mask, 1.0, t_uniform).reshape(-1, 1)
            else:
                t = t_uniform.reshape(-1, 1)
        # modified the time schedule
        # t = jnp.sqrt(1 - (t - 1)**2) # add by hiccup 
        
        # === Process the logic of inv_actions ===
        if self.config.get("inv_actions", False):
            # Map the action from [-1, 1] to the ℝ space.
            actions_y = inv_softsign(batch['actions'], self.config.get("scale_norm_factor", 0.18))
        else:
            # Keep the original movement.
            actions_y = batch['actions']
        # jax.debug.print("{}",actions_y)
        # Meanflow Training
        e = self.sample_noise(noise_rng, actions_y.shape)  
        # Flow process
        z = (1 - t) * actions_y + t * e 
        v = e - actions_y
        # JVP to calculate dgdt
        gn = self.network.select('actor_bc_flow')
        g, dgdt = jax.jvp(
            lambda args: gn(batch['observations'], args[0], args[1], params=grad_params),
            ((z, t),),
            ((v, jnp.ones_like(t)),)
        )
        # Loss function
        
        # g_tgt = 2*jnp.ones_like(v) - v - t * dgdt
        # g_tgt = 2*t - v - t * dgdt
        # g_tgt = 2*e*t - v - t * dgdt
        # g_tgt = e - v - t * dgdt
        # g_tgt = 2*z-v+2*v*t - t * dgdt
        g_tgt = v - t * dgdt
        # g_tgt = z + (t-1)*v - t * dgdt
        
        g_tgt = jax.lax.stop_gradient(g_tgt)
        # g_tgt = jnp.clip(g_tgt, -5, 5) # add by wzy.
        err = g - g_tgt
        
        # jax.debug.print("error = {}",err)
        
        mean_flow_loss = self.adaptive_l2_loss(err, t, mode="normal") # *  (t * (1.0 - t) + 0.25*jnp.ones_like(t)).mean() # Controlled by adaptive gamma. 

        consistency_loss = self.consistency_loss(batch, grad_params, rng)
        
        flow_loss = mean_flow_loss + consistency_loss * self.config.get('consistency_alpha', 1)
        

        return flow_loss, {
            'mean_flow_loss': mean_flow_loss,
            'consistency_loss': consistency_loss,
            'flow_loss': flow_loss,
            # 't_mean': t.mean(),
            # 't_var': t.var(),
            # 't_std': t.std(),
            # 't_min': t.min(),
            # 't_max': t.max(),
            # 't_median': jnp.median(t),
        }

    def consistency_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng = jax.random.split(rng, 2)
        t1, t2  = self.sample_discrete_t(rng, batch_size, time_steps=self.config.get("time_steps", 50))
        
        # === Process the logic of inv_actions ===
        if self.config.get("inv_actions", False):
            # Map the action from [-1, 1] to the ℝ space.
            actions_y = inv_softsign(batch['actions'], self.config.get("scale_norm_factor", 0.18))
        else:
            # Keep the original action.
            actions_y = batch['actions']
        
        # Consistency
        e = self.sample_noise(noise_rng, actions_y.shape) 
        # Flow 
        z_t1 = (1 - t1) * actions_y + t1 * e
        z_t2 = (1 - t2) * actions_y + t2 * e
        z_0_t1 = z_t1 - t1 * (z_t1 - self.network.select('actor_bc_flow')(batch['observations'], z_t1, t1, params=grad_params))
        # No grad. 
        z_0_t2 = z_t2 - t2 * (z_t2 - self.network.select('actor_bc_flow')(batch['observations'], z_t2, t2))
        z_0_t2 = jax.lax.stop_gradient(z_0_t2)
        
        consistency_loss = jnp.square(z_0_t1 - z_0_t2).mean()

        return consistency_loss



    
    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape

        # Predict action
        rng, noise_rng = jax.random.split(rng)
        # r_pred = jnp.zeros((batch_size, 1))
        t_pred = jnp.ones((batch_size, 1))
        noises = self.sample_noise(noise_rng, (batch_size, action_dim))  
        
        actions_y =  self.network.select('actor_bc_flow')(batch['observations'], noises, t = t_pred, params=grad_params)
        
        # === Process the logic of inv_actions ===
        if self.config.get("inv_actions", False):
            # Map the action from [-1, 1] to the ℝ space
            actions = jax.nn.soft_sign(actions_y) / (1 - self.config.get("scale_norm_factor", 0.18))
        else:
            # Keep the original action.
            actions = actions_y
        
        # Add bound_loss
        upper_bound = jnp.ones_like(actions)
        lower_bound = -jnp.ones_like(actions)
        bound_loss = jnp.mean(nn.relu(actions - upper_bound)) + jnp.mean(nn.relu(lower_bound - actions))

        actions = jnp.clip(actions, -1, 1)

        # Calculate 
        qs = self.network.select('critic')(batch['observations'], actions=actions)
        q = jnp.mean(qs, axis=0)
        q_loss = -q.mean()
        
        if self.config["normalize_q_loss"]:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        actor_loss =  q_loss   + bound_loss * self.config.get('bound_loss_weight', 1) # Add bound loss 
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bound_loss': bound_loss, 
            'q': q.mean(),
            'mse': mse,
        }
    
    def sample_t_r(self, t_rng, r_rng, batch_size, flow_ratio=0.0):
        """
        This function is used in the initial meanflow, not used in this file. 
        """
        # Generate two random samples.  
        samples = jax.random.uniform(t_rng, (batch_size, 2))
        
        # Set r<=t
        t = jnp.maximum(samples[:, 0:1], samples[:, 1:2])
        r = jnp.minimum(samples[:, 0:1], samples[:, 1:2])
        
        if flow_ratio > 0:
            indices_key = jax.random.fold_in(r_rng, 0)
            indices = jax.random.permutation(indices_key, jnp.arange(batch_size))
            
            num_selected = int(flow_ratio * batch_size)
            selected_indices = indices[:num_selected]
            
            mask = jnp.zeros(batch_size, dtype=bool)
            mask = mask.at[selected_indices].set(True)
            mask = jnp.reshape(mask, (-1, 1))
            
            r = jnp.where(mask, t, r)
            
        return t, r
    


    @jax.jit
    def total_loss(self, batch, grad_params, rng, current_step=0):
        """Calculate total loss"""
        info = {}
        rng = rng if rng is not None else self.rng

        meanflow_rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        # Critic network
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Meanflow network
        flow_loss, meanflow_info = self.meanflow_loss(batch, grad_params, meanflow_rng)
        for k, v in meanflow_info.items():
            info[f'meanflow/{k}'] = v

        # Actor network
        actor_loss, actor_info = self.actor_loss(
            batch, grad_params, actor_rng
        )
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        
        total_loss = critic_loss + actor_loss + flow_loss * self.config.get('alpha', 1.0)
         

        info["total_loss"] = total_loss
        return total_loss, info

    def target_update(self, network, module_name):
        """ EMA update. """
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, current_step=0):
        """Update the full parameter the all the network."""
        new_rng, rng = jax.random.split(self.rng)
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng, current_step=current_step)
        
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
               
        # Use the simplified target_update (no return value needed)
        self.target_update(new_network, 'critic')
        
        # Log the current metrics with separate learning rates for actor and critic
        if current_step is not None:
            # Record actor learning rate
            actor_lr_schedule = self.config.get('actor_lr_schedule')
            if actor_lr_schedule is not None:
                current_actor_lr = actor_lr_schedule(current_step)
                info['metrics/actor_learning_rate'] = current_actor_lr
            
            # Record critic learning rate
            critic_lr_schedule = self.config.get('critic_lr_schedule')
            if critic_lr_schedule is not None:
                current_critic_lr = critic_lr_schedule(current_step)
                info['metrics/critic_learning_rate'] = current_critic_lr
            
            # Keep backward compatibility
            lr_schedule = self.config.get('lr_schedule')
            if lr_schedule is not None:
                current_lr = lr_schedule(current_step)
                info['metrics/learning_rate'] = current_lr
                
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def pretrain(self, batch, current_step=None):
        """ Pretrain the meanflow part to let it adapt to the action spaces"""
        new_rng, rng = jax.random.split(self.rng)
        
        def pretrain_loss(grad_params):
            return self.meanflow_loss(
                batch, grad_params, rng=rng
            )
        new_network, info = self.network.apply_loss_fn(loss_fn=pretrain_loss)
        
        # Log learning rates during pretraining
        if current_step is not None:
            # Record actor learning rate (main focus during pretraining)
            actor_lr_schedule = self.config.get('actor_lr_schedule')
            if actor_lr_schedule is not None:
                current_actor_lr = actor_lr_schedule(current_step)
                info['metrics/actor_learning_rate'] = current_actor_lr
            
            # Record critic learning rate for completeness
            critic_lr_schedule = self.config.get('critic_lr_schedule')
            if critic_lr_schedule is not None:
                current_critic_lr = critic_lr_schedule(current_step)
                info['metrics/critic_learning_rate'] = current_critic_lr
            
            # Keep backward compatibility
            lr_schedule = self.config.get('lr_schedule')
            if lr_schedule is not None:
                current_lr = lr_schedule(current_step)
                info['metrics/learning_rate'] = current_lr
                
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        temperature=1,
        seed=None,
    ):
        """ Generate action from gn """
        action_seed, noise_seed = jax.random.split(seed)
        action_shape = (
            *observations.shape[: -len(self.config['ob_dims'])],
            self.config['action_dim'],
        )
        e = self.sample_noise(action_seed, action_shape)
        
        batch_size = observations.shape[0]
        # r = jnp.zeros((batch_size, 1))
        t = jnp.ones((batch_size, 1))
        # Generate action from g(x_t, t)
        if self.config['encoder'] is not None:
            encoded_obs = self.network.select('actor_bc_flow_encoder')(observations)
            actions_y = e - (self.network.select('actor_bc_flow')(encoded_obs, e,  t, is_encoded=True) - 2*jnp.ones_like(e))
        else:
            # actions_y = (self.network.select('actor_bc_flow')(observations, e,  t)) - e
            # # actions = e + (self.network.select('actor_bc_flow')(observations, e,  t)- t)
            # # actions = self.network.select('actor_bc_flow')(observations, e,  t)
            actions_y = e - self.network.select('actor_bc_flow')(observations, e,  t)
            # actions_y = self.network.select('actor_bc_flow')(observations, e,  t)

        # Process the logic of inv_actions ===
        if self.config.get("inv_actions", False):
            # Map the action from [-1, 1] to the ℝ space.
            actions = jax.nn.soft_sign(actions_y) / (1 - self.config.get("scale_norm_factor", 0.18))
        else:
            # Keep the original action.
            actions = actions_y

        actions = jnp.clip(actions, -1, 1)

        return actions


    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        import copy  
        
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        # Create input examples. 
        batch_size = ex_observations.shape[0]
        ex_t = jnp.ones((batch_size, 1))  
        ex_r = jnp.zeros((batch_size, 1))  
        
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
    
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
    
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        
        actor_bc_flow_def = MFDiT_SIM_e(
            # input_dim=action_dim, 
            hidden_dim=config['actor_hidden_dims'],
            depth=config['actor_depth'],
            num_heads=config['actor_num_heads'],
            output_dim=action_dim,  
            encoder=encoders.get('actor_bc_flow'),
            tanh_squash = config['tanh_squash'],
            use_output_layernorm = config["use_output_layernorm"],
        )
    
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_t)),
        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        
        # Basic learning rate setup
        base_lr = config['lr']
        # Import important factors
        from absl import flags
        FLAGS = flags.FLAGS
        pretrain_steps = FLAGS.offline_steps * FLAGS.pretrain_factor
        q_learning_steps = FLAGS.offline_steps + FLAGS.online_steps
        total_steps = pretrain_steps + q_learning_steps

        decay_steps = total_steps
        min_lr = base_lr * config.get('lr_min_ratio', 0.1)  
        warmup_steps = int(total_steps*0.05)
        
        if warmup_steps > 0:
            actor_lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=base_lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=min_lr
            )
        else:
            actor_lr_schedule = optax.cosine_decay_schedule(
                init_value=base_lr,
                decay_steps=decay_steps,
                alpha=config.get('lr_min_ratio', 0.1)
            )
        
        critic_lr_schedule = lambda _: base_lr*3
        
        config['actor_lr_schedule'] = actor_lr_schedule
        config['critic_lr_schedule'] = critic_lr_schedule
        
        def create_mask_fn(prefix):
            def mask_fn(params):
                flat_params = {'/'.join(k): v for k, v in flax.traverse_util.flatten_dict(params).items()}
                return {k: k.startswith(f'modules_{prefix}') for k in flat_params.keys()}
            return mask_fn
        
        critic_tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=critic_lr_schedule)
        )
        
        actor_tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=actor_lr_schedule)
        )
        
        def param_partition(params):
            flat_params = flax.traverse_util.flatten_dict(params)
            partition = {}
            for key_tuple in flat_params.keys():
                key_str = '/'.join(key_tuple)
                if key_str.startswith('modules_critic') or key_str.startswith('modules_target_critic'):
                    partition[key_tuple] = 'critic'
                elif key_str.startswith('modules_actor_bc_flow_encoder'):
                    partition[key_tuple] = 'actor'
                elif key_str.startswith('modules_actor_bc_flow'):
                    partition[key_tuple] = 'actor'
                else:
                    partition[key_tuple] = 'actor'
            # Ensure to return a regular Python dict instead of a FrozenDict.
            unflattened = flax.traverse_util.unflatten_dict(partition)
            return jax.tree_map(lambda x: x, unflattened)  # Convert to a normal dict

        network_params = network_def.init(init_rng, **network_args)['params']
        # Add target_critic params
        from flax.core import FrozenDict, unfreeze, freeze
        network_params_dict = unfreeze(network_params)
        network_params_dict['modules_target_critic'] = copy.deepcopy(network_params_dict['modules_critic'])

        # Generate parameter partition mask - Ensure to use a regular dictionary.
        param_labels = param_partition(dict(network_params_dict))  # Explicit conversion to dict

        # Create multi_transform using the correct mask tree.
        network_tx = optax.multi_transform(
            {
                'critic': critic_tx,
                'actor': actor_tx,
            },
            param_labels  # Pass in the actual mask tree instead of the function.
        )
        
        # Then initialize the optimizer state - make sure to use a regular dict.
        opt_state = network_tx.init(dict(network_params_dict))  # Explicit conversion to dict
        
        # Create TrainState (using FrozenDict)
        network_params_frozen = freeze(network_params_dict)
        network = TrainState(
            step=1,
            apply_fn=network_def.apply,
            model_def=network_def,
            params=network_params_frozen,
            tx=network_tx,
            opt_state=opt_state
        )

        # Remove this line since we've already added modules_target_critic
        # params = network.params
        # network = network.replace(params=params.copy(add_or_replace={'modules_target_critic': params['modules_critic']}))

        if 'metric' not in config:
            config['metric'] = lambda x: jnp.mean(x ** 2)
            
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


    def adaptive_l2_loss(self, error,t, gamma=None, c=None, mode="normal"):
        gamma = gamma if gamma is not None else self.config.get('adaptive_gamma', 0.5)
        c = c if c is not None else self.config.get('adaptive_c', 1e-3)
        
        delta_sq = jnp.mean(error ** 2, axis=-1)
        delta_sq = jnp.maximum(delta_sq, 1e-12)
        
        p = 1.0 - gamma
        denominator = jnp.power(delta_sq + c, p)
        denominator = jnp.maximum(denominator, 1e-12)  
        w = 1.0 / denominator
        # jax.debug.print("adaptive_l2_loss: w = {}",w)
        w = jnp.clip(w, 1e-6, 1e6) 
        loss = delta_sq
        if mode!="normal":
            time_factor = (t * (1.0 - t) + 0.75).squeeze(-1)
            w = w * time_factor
        
        
        return jnp.mean(jax.lax.stop_gradient(w) * loss)

    def sample_discrete_t(self, rng, batch_size, time_steps=100):
            """Sample discrete time steps t and t_con.
            
            Args:
                rng: Random number generator
                batch_size: Batch size
                time_steps: Number of time steps
                
            Returns:
                t, t_con: Time samples with shape (batch_size, 1)
            """
            t_rng, t_con_rng = jax.random.split(rng)
            
            # Create evenly spaced time step values (from 1/time_steps to 1, with interval 1/time_steps)
            time_values = jnp.linspace(1/time_steps, 1.0, time_steps)
            
            # Randomly select time step indices for each sample in the batch
            t_indices = jax.random.randint(t_rng, (batch_size,), 0, time_steps)
            t_con_indices = jax.random.randint(t_con_rng, (batch_size,), 0, time_steps)
            
            # Get corresponding time step values based on indices and reshape to (batch_size, 1)
            t1 = time_values[t_indices].reshape(-1, 1)
            t2 = time_values[t_con_indices].reshape(-1, 1)
            
            return t1, t2

    def get_param_count(self):
        """Calculate and return the number of parameters in the network.
        
        Returns:
            dict: Dictionary containing parameter counts for each network module and total parameter count
        """
        param_counts = {}
        total_params = 0
        
        # Iterate through all network modules
        for module_name, module_params in self.network.params.items():
            if isinstance(module_params, dict):
                # Calculate parameter count for this module
                module_count = 0
                for param in jax.tree_util.tree_leaves(module_params):
                    param_size = param.size
                    module_count += param_size
                
                param_counts[module_name] = module_count
                total_params += module_count
        
        param_counts['total'] = total_params
        return param_counts

    def print_param_stats(self):
        """Print network parameter statistics."""
        param_counts = self.get_param_count()
        
        print("MeanNormFQLAgent Network Parameter Statistics:")
        print("-" * 40)
        
        # Print parameter count for each module
        for module_name, count in param_counts.items():
            if module_name != 'total':
                print(f"{module_name}: {count:,} parameters ({count * 4 / (1024**2):.2f} MB)")
        
        # Print total parameter count
        total = param_counts['total']
        print("-" * 40)
        print(f"Total parameters: {total:,} ({total * 4 / (1024**2):.2f} MB)")

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='meanflowql_beta',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  
            lr_min_ratio=0.05,  
            batch_size=256,  # Batch size.
            actor_hidden_dims=256,  # Actor network hidden dimensions.
            actor_depth=3, # Transformer depth
            actor_num_heads = 2 ,# Transformer num heads
            flow_ratio = 0.3,  # Control the rate of r==t, but we set r=0 in our experiments. 
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            q_agg='mean',  # Aggregation method for target Q values.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            tanh_squash=False,  # Whether to use tanh activation for the actor.
            use_output_layernorm=False,
            time_steps =  50,  # Discrete time steps. 
            t_one_prob = 0.0,  # Probability of setting t=1 (0.0-1.0)
            sigma=1.0,  # Noise scale
            consistency_alpha = 0.0,  # Consistency loss weight. 
            adaptive_gamma=0.5,  # This parameter is used for controlling the loss function of meanflow.
            adaptive_c=1e-3,     # Control the loss function of meanflow. 
            bound_loss_weight=0.0, # Control the bound_loss weight.
            noise_type = "gaussian",  # The noise type, it can be uniform and gaussian. 
            scale_norm_factor=0.99, # used in the the normalization in action sequence. 
            inv_actions = True,  # Whether to use inverse actions
        )
    )
    return config
