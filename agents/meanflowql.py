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
from utils.dit_jax import MFDiT, MFDiT_SIM


class MeanFlowQL_Agent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    current_alpha: float = 1.0  # Current alpha state
    loss_history: jnp.ndarray = None  # Initialize as fixed shape in __post_init__
    valid_count: int = 0  # Track valid history count
    
    def __post_init__(self):
        if self.loss_history is None:
            # Initialize as fixed shape zero array
            history_window_size = self.config.get('loss_history_window_size', 5)
            object.__setattr__(self, 'loss_history', jnp.zeros(history_window_size))

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

        # Calculate half of num_candidates, ensure it's at least 1
        half_candidates = jnp.maximum(1, self.config['num_candidates'] // 2)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng, num_candidates=half_candidates)
        
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
        # Discrete time schedule
        time_steps = self.config.get('time_steps', 10)
        if time_steps<=1000:
            time_values = jnp.linspace(1/time_steps, 1.0, time_steps)
            indices = jax.random.randint(t_rng, (batch_size,), 0, time_steps)
            t = time_values[indices].reshape(-1, 1)
        else:
            t = jax.random.uniform(t_rng, (batch_size, 1))
        # modified the time schedule
        actions = batch['actions']
        # Meanflow Training
        e = self.sample_noise(noise_rng, batch['actions'].shape)  
        # Flow process
        z = (1 - t) * actions + t * e 
        v = e - actions
        # JVP to calculate dgdt
        gn = self.network.select('actor_bc_flow')
        g, dgdt = jax.jvp(
            lambda args: gn(batch['observations'], args[0], args[1], params=grad_params),
            ((z, t),),
            ((v, jnp.ones_like(t)),)
        )
        # Loss function
        g_tgt = z + (t-1)*v - t * dgdt
        g_tgt = jax.lax.stop_gradient(g_tgt)
        g_tgt = jnp.clip(g_tgt, -5, 5) 
        err = g - g_tgt
        mean_flow_loss = self.adaptive_l2_loss(err, t, mode="normal") 

        consistency_loss = self.consistency_loss(batch, grad_params, rng)
        
        flow_loss = mean_flow_loss + consistency_loss * self.config.get('consistency_alpha', 1)
        

        return flow_loss, {
            'mean_flow_loss': mean_flow_loss,
            'consistency_loss': consistency_loss,
            'flow_loss': flow_loss,
        }

    def consistency_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng = jax.random.split(rng, 2)
        t1, t2  = self.sample_discrete_t(rng, batch_size, time_steps=self.config.get("time_steps", 50))
        # Consistency
        actions = batch['actions']
        e = self.sample_noise(noise_rng, batch['actions'].shape) 
        # Flow 
        z_t1 = (1 - t1) * actions + t1 * e
        z_t2 = (1 - t2) * actions + t2 * e
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
        t_pred = jnp.ones((batch_size, 1))
        noises = self.sample_noise(noise_rng, (batch_size, action_dim))  
        
        actions =  self.network.select('actor_bc_flow')(batch['observations'], noises, t = t_pred, params=grad_params)
        
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
        """Calculate total loss with alpha weight scheduling"""
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
        
        # Get alpha scheduling configuration
        use_dynamic_alpha = self.config.get('use_dynamic_alpha', False)
        alpha_update_interval = self.config.get('alpha_update_interval', 200)
        
        # Use JAX operations instead of Python boolean operators
        step_mod_interval = current_step % alpha_update_interval
        should_update_alpha = jnp.logical_and(
            step_mod_interval == 0,
            current_step > 0
        )
        
        # Calculate 20% of total training steps as warmup threshold for recording only without adjustment
        max_training_steps = self.config.get('pretrain_plus_offline_steps', 1000000)
        warmup_steps = int(max_training_steps * 0.1)  # First 20% steps for recording only without adjustment
        
        # Check if we are in the first 20% of training steps
        is_in_warmup = current_step < warmup_steps
        
        # Use current alpha value during warmup or when no update is needed
        should_calculate_dynamic_alpha = jnp.logical_and(
            should_update_alpha,
            jnp.logical_not(is_in_warmup)
        )
        
        # Calculate alpha weight based on schedule type - only compute what's needed
        if use_dynamic_alpha:
            # Only calculate dynamic alpha when needed (on update steps and not in warmup)
            alpha_weight = jnp.where(
                should_calculate_dynamic_alpha,
                self._calculate_dynamic_alpha(meanflow_info['mean_flow_loss'], current_step, info),
                self.current_alpha  # Use cached current_alpha when not updating or in warmup
            )
        else:
            # Only calculate cosine alpha for non-dynamic scheduling
            alpha_weight = self._calculate_cosine_alpha(current_step)
        
        total_loss = critic_loss + actor_loss + flow_loss * alpha_weight
        
        # Log the current alpha weight for monitoring
        info["alpha_weight"] = alpha_weight
        info["total_loss"] = total_loss
        return total_loss, info
    
    def _calculate_cosine_alpha(self, current_step):
        """Calculate alpha weight using cosine decay schedule"""
        max_training_steps = self.config.get('pretrain_plus_offline_steps', 1000000)
        initial_alpha = self.config.get('alpha', 1.0)
        min_alpha = initial_alpha * 0.1  # Dynamic calculation: alpha * 0.1
        
        # Cosine decay schedule: starts at initial_alpha, decays to min_alpha
        progress = jnp.minimum(current_step / max_training_steps, 1.0)
        alpha_weight = min_alpha + (initial_alpha - min_alpha) * 0.5 * (1 + jnp.cos(jnp.pi * progress))
        
        return alpha_weight
    

    def _calculate_dynamic_alpha(self, mean_flow_loss, current_step, info):
        """Calculate alpha weight based on mean_flow_loss dynamics with sliding window average"""
        # Get configuration parameters with dynamic calculation
        initial_alpha = self.config.get('alpha', 1.0)
        min_alpha = initial_alpha * 0.1  # Dynamic calculation: alpha * 0.1
        max_alpha = initial_alpha * 10.0  # Dynamic calculation: alpha * 10
        
        # Dynamic adjustment parameters
        loss_multiplier_threshold = self.config.get('loss_multiplier_threshold', 5)
        alpha_increase_factor = self.config.get('alpha_increase_factor', 1.2)
        alpha_decrease_factor = self.config.get('alpha_decrease_factor', 0.8)
        history_window_size = self.config.get('loss_history_window_size', 20)
        
        base_alpha = self.current_alpha
        
        # Use valid_count to check if enough history data is available
        has_enough_history = self.valid_count >= history_window_size
        
        # Calculate average of valid history data using mask instead of dynamic slicing
        effective_count = jnp.minimum(self.valid_count, history_window_size)
        
        # Create a mask to sum only valid historical data
        mask = jnp.arange(history_window_size) < effective_count
        masked_history = self.loss_history * mask
        
        # Calculate historical average loss using mask
        historical_avg_loss = jnp.where(
            effective_count > 0,
            jnp.sum(masked_history) / effective_count,
            0.0
        )
        
        # Conditionally calculate alpha adjustment
        should_increase = jnp.logical_and(
            effective_count >= 2,  # Need at least 2 data points for comparison
            mean_flow_loss > (historical_avg_loss * loss_multiplier_threshold)
        )
        should_decrease = jnp.logical_and(
            effective_count >= 2,
            mean_flow_loss < (historical_avg_loss / loss_multiplier_threshold)
        )
        
        alpha_weight = jnp.where(
            should_increase,
            jnp.minimum(base_alpha * alpha_increase_factor, max_alpha),
            base_alpha
        )
        alpha_weight = jnp.where(
            should_decrease,
            jnp.maximum(alpha_weight * alpha_decrease_factor, min_alpha),
            alpha_weight
        )
    
        info['alpha/historical_avg_loss'] = historical_avg_loss
        info['alpha/valid_count'] = self.valid_count
        info['alpha/current_vs_historical_ratio'] = jnp.where(
            effective_count > 0,
            mean_flow_loss / jnp.maximum(historical_avg_loss, 1e-8),
            1.0
        )
        # Add dynamic bounds info for debugging
        info['alpha/min_alpha_bound'] = min_alpha
        info['alpha/max_alpha_bound'] = max_alpha
        
        return alpha_weight

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
               
        
        self.target_update(new_network, 'critic')

        use_dynamic_alpha = self.config.get('use_dynamic_alpha', False) 

        # Update current_alpha if dynamic alpha is used
        if use_dynamic_alpha:
            max_training_steps = self.config.get('pretrain_plus_offline_steps', 1000000)
            warmup_steps = int(max_training_steps * 0.1) 
            is_in_warmup = current_step < warmup_steps
            
            alpha_update_interval = self.config.get('alpha_update_interval', 200)
            step_mod_interval = current_step % alpha_update_interval
            should_update_alpha = jnp.logical_and(
                step_mod_interval == 0,
                current_step > 0
            )
            
            # Get current mean flow loss for history tracking
            current_loss = info.get('meanflow/mean_flow_loss', 0.0)
            
            # Use fixed shape sliding window update
            history_window_size = self.config.get('loss_history_window_size', 5)

            new_loss_history = jnp.where(
                should_update_alpha,
                jnp.roll(self.loss_history, -1).at[-1].set(current_loss),
                self.loss_history
            )
            
            # Update valid count
            new_valid_count = jnp.where(
                should_update_alpha,
                jnp.minimum(self.valid_count + 1, history_window_size),
                self.valid_count
            )
            
            new_alpha = jnp.where(
                is_in_warmup,
                self.current_alpha, 
                info.get('alpha_weight', self.current_alpha)  
            )
            
            updated_agent = self.replace(
            network=new_network, 
            rng=new_rng, 
            current_alpha=new_alpha,
            loss_history=new_loss_history,
            valid_count=new_valid_count
            )
        else:
            updated_agent = self.replace(network=new_network, rng=new_rng)
        
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
                
        return updated_agent, info

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
    def sample_actions_best(
        self,
        observations,
        temperature=1,
        seed=None,
        num_candidates=None,
    ):
        """Optimized version with early observation expansion for better parallelization"""
        action_seed, noise_seed = jax.random.split(seed)
        
        # Handle single sample input when using encoder
        if self.config['encoder'] is not None and observations.ndim == 3:
            observations = observations[None, :]
        
        original_batch_size = observations.shape[0]
        action_dim = self.config['action_dim']
        # Use provided num_candidates or fall back to config value
        num_candidates = num_candidates if num_candidates is not None else self.config['num_candidates']
        
        # FIXED: Convert to concrete integer for shape operations
        # Use jax.lax.convert_element_type to ensure it's a concrete value
        num_candidates_concrete = int(num_candidates) if isinstance(num_candidates, (int, float)) else int(self.config['num_candidates'])
        
        # Early expansion: expand observations at the beginning
        # This creates a larger batch (original_batch_size * num_candidates)
        # Shape: (num_candidates, batch_size, *obs_dims) -> (num_candidates * batch_size, *obs_dims)
        obs_expanded = jnp.tile(observations[None, :], (num_candidates_concrete, 1) + (1,) * (observations.ndim - 1))
        obs_flat = obs_expanded.reshape(-1, *observations.shape[1:])
        expanded_batch_size = obs_flat.shape[0]  # num_candidates * original_batch_size
        
        # Generate candidate seeds and noise for the expanded batch
        candidate_seeds = jax.random.split(action_seed, num_candidates_concrete)
        
        # Generate noise for each candidate group
        noise_fn = lambda seed: self.sample_noise(seed, (original_batch_size, action_dim))
        all_noise = jax.vmap(noise_fn)(candidate_seeds)  # Shape: (num_candidates, batch_size, action_dim)
        noise_flat = all_noise.reshape(-1, action_dim)  # Shape: (num_candidates * batch_size, action_dim)
        
        # Time values for the expanded batch
        t_flat = jnp.ones((expanded_batch_size, 1))
        
        # Generate candidate actions using the expanded batch
        if self.config['encoder'] is not None:
            # Encode all observations at once (more efficient)
            encoded_obs_flat = self.network.select('actor_bc_flow_encoder')(obs_flat)
            
            # Handle different encoded_obs dimensions
            if encoded_obs_flat.ndim == 3:
                encoded_obs_flat = encoded_obs_flat[:, -1, :]  # Take last timestep if 3D
            
            # Generate actions using encoded observations
            candidate_actions_flat = self.network.select('actor_bc_flow')(
                encoded_obs_flat, noise_flat, t_flat, is_encoded=True
            )
        else:
            # Generate actions using raw observations
            candidate_actions_flat = self.network.select('actor_bc_flow')(
                obs_flat, noise_flat, t_flat
            )
        
        # Clip actions
        candidate_actions_flat = jnp.clip(candidate_actions_flat, -1, 1)
        
        # Evaluate Q-values for all candidate actions at once
        # Use original observations for critic evaluation (critic has its own encoder)
        q_values_flat = self.network.select('target_critic')(obs_flat, actions=candidate_actions_flat)
        
        # Handle ensemble dimension efficiently - always use mean aggregation
        if q_values_flat.ndim > 1:
            q_values_flat = jnp.mean(q_values_flat, axis=-1 if q_values_flat.shape[-1] == 2 else 0)
        
        # Reshape back to (num_candidates, original_batch_size)
        q_values = q_values_flat.reshape(num_candidates_concrete, original_batch_size)
        candidate_actions = candidate_actions_flat.reshape(num_candidates_concrete, original_batch_size, action_dim)
        
        # Select best actions for each sample in the original batch
        best_indices = jnp.argmax(q_values, axis=0)
        best_actions = candidate_actions[best_indices, jnp.arange(original_batch_size)]
        
        return best_actions

    @jax.jit
    def sample_actions_mean(
        self,
        observations,
        temperature=1,
        seed=None,
        num_candidates=None,
    ):
        """ Generate action from gn with averaging over N candidates """
        action_seed, noise_seed = jax.random.split(seed)
        
        # Handle single sample input when using encoder
        if self.config['encoder'] is not None and observations.ndim == 3:
            # Add batch dimension: (64,64,3) -> (1,64,64,3)
            observations = observations[None, :]
        
        batch_size = observations.shape[0]
        action_dim = self.config['action_dim']
        
        # Use provided num_candidates or fall back to config value
        num_candidates = num_candidates if num_candidates is not None else self.config['num_candidates']
        
        # Split seeds for each candidate
        candidate_seeds = jax.random.split(action_seed, num_candidates)
        
        # Vectorized noise generation - expand batch dimension
        # Shape: (num_candidates, batch_size, action_dim)
        expanded_action_shape = (
            num_candidates,
            *observations.shape[: -len(self.config['ob_dims'])],
            self.config['action_dim'],
        )
        
        # Generate all noise at once using vmap
        noise_fn = lambda seed: self.sample_noise(seed, expanded_action_shape[1:])
        all_noise = jax.vmap(noise_fn)(candidate_seeds)  # Shape: (num_candidates, batch_size, action_dim)
        
        # Shape: (num_candidates, batch_size, 1)
        t_expanded = jnp.ones((num_candidates, batch_size, 1))
        
        # Vectorized action generation
        if self.config['encoder'] is not None:
            # Encode observations once
            encoded_obs = self.network.select('actor_bc_flow_encoder')(observations)
            
            # Handle different encoded_obs dimensions robustly
            if encoded_obs.ndim == 1:
                # Convert (512,) to (1, 512) for consistency
                encoded_obs = encoded_obs[None, :]
            elif encoded_obs.ndim == 3:
                # Handle (5, 256, 512) or (5, 1, 512) cases
                # Option 1: Take the last timestep
                encoded_obs = encoded_obs[:, -1, :]  # (5, 512)
            # encoded_obs.ndim == 2 case is already handled (no change needed)
            
            # Now encoded_obs is guaranteed to be 2D: (batch_size, feature_dim)
            # Expand for vectorized computation
            encoded_obs_expanded = jnp.tile(encoded_obs[None, :, :], (num_candidates, 1, 1))
            
            def generate_action(encoded_obs, noise, t):
                # encoded_obs is already (batchsize, 512) after vmap
                return self.network.select('actor_bc_flow')(encoded_obs, noise, t, is_encoded=True)
            
            candidate_actions = jax.vmap(generate_action)(encoded_obs_expanded, all_noise, t_expanded)
        else:
            # Expand observations for vectorized computation (only when not using encoder)
            # Shape: (num_candidates, batch_size, obs_dim)
            obs_expanded = jnp.tile(observations[None, :, :], (num_candidates, 1, 1))
            
            # Vectorized flow computation without encoder
            def generate_action(obs, noise, t):
                return self.network.select('actor_bc_flow')(obs, noise, t)
            
            # Use vmap to process all candidates at once
            candidate_actions = jax.vmap(generate_action)(obs_expanded, all_noise, t_expanded)
        
        # Ensure candidate_actions has correct shape: (num_candidates, batch_size, action_dim)
        if candidate_actions.ndim == 4:  # (num_candidates, batch_size, 1, action_dim)
            candidate_actions = candidate_actions.squeeze(2)
        elif candidate_actions.ndim > 4:
            # Reshape to correct dimensions
            candidate_actions = candidate_actions.reshape(num_candidates, batch_size, -1)
        

        # Average all candidate actions instead of selecting best one
        # Shape: (num_candidates, batch_size, action_dim) -> (batch_size, action_dim)
        averaged_actions = jnp.mean(candidate_actions, axis=0)
        # Clip actions - FIXED: clip averaged_actions instead of candidate_actions
        averaged_actions = jnp.clip(averaged_actions, -1, 1)
        
        return averaged_actions

    @jax.jit
    def sample_actions_normal(
        self,
        observations,
        temperature=1,
        seed=None,
    ):
        """ Generate action from gn (the simplest way.) """
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
            actions = self.network.select('actor_bc_flow')(encoded_obs, e,  t, is_encoded=True)
        else:
            actions = self.network.select('actor_bc_flow')(observations, e,  t)

        actions = jnp.clip(actions, -1, 1)

        return actions

    @jax.jit
    def sample_actions(
        self,
        observations,
        temperature=1,
        seed=None,
        num_candidates=None,
    ):
        """ Generate action based on action_mode config parameter """
        action_mode = self.config.get('action_mode', 'best')  # Default to 'best' if not specified
        
        if action_mode == 'mean':
            return self.sample_actions_mean(observations, temperature, seed, num_candidates)
        elif action_mode == 'best':
            return self.sample_actions_best(observations, temperature, seed, num_candidates)
        elif action_mode == "normal":
            return self.sample_actions_normal(observations, temperature, seed)
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}. Must be 'best' or 'mean'.")

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
        
        actor_bc_flow_def = MFDiT_SIM(
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
        offline_steps = FLAGS.offline_steps
        online_steps = FLAGS.online_steps
        total_steps = pretrain_steps + offline_steps + online_steps

        # Calculate pretrain_plus_offline_steps and add to config
        pretrain_plus_offline_steps = pretrain_steps + offline_steps
        config['pretrain_plus_offline_steps'] = pretrain_plus_offline_steps

        # Calculate phase boundaries
        offline_end_step = pretrain_steps + offline_steps
        
        min_lr = base_lr * config.get('lr_min_ratio', 0.05)  
        warmup_steps = int((pretrain_steps + offline_steps) * 0.05)  # Warmup only for offline phase
        
        # Create custom learning rate schedule for offline + online phases
        def create_phase_aware_lr_schedule(base_lr, min_lr, warmup_steps, offline_end_step):
            def lr_schedule(step):
                # Phase 1: Offline training (pretrain + offline) - use cosine decay
                offline_phase_condition = step <= offline_end_step
                
                # For offline phase: warmup + cosine decay
                if warmup_steps > 0:
                    # Warmup phase
                    warmup_condition = step <= warmup_steps
                    warmup_lr = (step / warmup_steps) * base_lr
                    
                    # Cosine decay phase (after warmup, within offline training)
                    cosine_progress = (step - warmup_steps) / jnp.maximum(offline_end_step - warmup_steps, 1)
                    cosine_progress = jnp.clip(cosine_progress, 0.0, 1.0)
                    cosine_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + jnp.cos(jnp.pi * cosine_progress))
                    
                    offline_lr = jnp.where(warmup_condition, warmup_lr, cosine_lr)
                else:
                    # No warmup, direct cosine decay for offline phase
                    cosine_progress = step / jnp.maximum(offline_end_step, 1)
                    cosine_progress = jnp.clip(cosine_progress, 0.0, 1.0)
                    offline_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + jnp.cos(jnp.pi * cosine_progress))
                
                # Phase 2: Online fine-tuning - use fixed min_lr
                online_lr = min_lr
                
                # Return appropriate learning rate based on current phase
                return jnp.where(offline_phase_condition, offline_lr, online_lr)
            
            return lr_schedule
        
        # Create phase-aware learning rate schedules
        actor_lr_schedule = create_phase_aware_lr_schedule(base_lr, min_lr, warmup_steps, offline_end_step)
        critic_lr_schedule = lambda _: 3e-4
        
        config['actor_lr_schedule'] = actor_lr_schedule
        config['critic_lr_schedule'] = critic_lr_schedule

        
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
            # 确保返回普通的 Python dict，而不是 FrozenDict
            unflattened = flax.traverse_util.unflatten_dict(partition)
            return jax.tree.map(lambda x: x, unflattened)  # 转换为普通 dict

        network_params = network_def.init(init_rng, **network_args)['params']
        # Add target_critic params
        from flax.core import FrozenDict, unfreeze, freeze
        network_params_dict = unfreeze(network_params)
        network_params_dict['modules_target_critic'] = copy.deepcopy(network_params_dict['modules_critic'])

        param_labels = param_partition(dict(network_params_dict))  

        network_tx = optax.multi_transform(
            {
                'critic': critic_tx,
                'actor': actor_tx,
            },
            param_labels  
        )

        opt_state = network_tx.init(dict(network_params_dict))  
        
        network_params_frozen = freeze(network_params_dict)
        network = TrainState(
            step=1,
            apply_fn=network_def.apply,
            model_def=network_def,
            params=network_params_frozen,
            tx=network_tx,
            opt_state=opt_state
        )


        if 'metric' not in config:
            config['metric'] = lambda x: jnp.mean(x ** 2)
            
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        initial_alpha = config.get('alpha', 1.0)
        history_window_size = config.get('loss_history_window_size', 20)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config), current_alpha=initial_alpha, loss_history=jnp.zeros(history_window_size))


    def adaptive_l2_loss(self, error,t, gamma=None, c=None, mode="normal"):
        gamma = gamma if gamma is not None else self.config.get('adaptive_gamma', 0.5)
        c = c if c is not None else self.config.get('adaptive_c', 1e-3)
        
        delta_sq = jnp.mean(error ** 2, axis=-1)
        delta_sq = jnp.maximum(delta_sq, 1e-12)
        
        p = 1.0 - gamma
        denominator = jnp.power(delta_sq + c, p)
        denominator = jnp.maximum(denominator, 1e-12)  
        w = 1.0 / denominator
        w = jnp.clip(w, 1e-6, 1e6) 
        loss = delta_sq
        if mode!="normal":
            time_factor = (t * (1.0 - t) + 0.75).squeeze(-1)
            w = w * time_factor
        
        
        return jnp.mean(jax.lax.stop_gradient(w) * loss)

    def sample_discrete_t(self, rng, batch_size, time_steps=100):
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
        """Calculate and return the number of parameters in the network."""
        params = self.network.params
        if hasattr(params, 'unfreeze'):
            params = params.unfreeze()
        
        param_counts = {}
        
        # Calculate module-wise parameter counts
        for module_name, module_params in params.items():
            module_leaves = jax.tree_util.tree_leaves(module_params)
            param_counts[module_name] = sum(param.size for param in module_leaves)
        
        # Calculate total parameters
        all_leaves = jax.tree_util.tree_leaves(params)
        param_counts['total'] = sum(param.size for param in all_leaves)
        
        return param_counts

    def print_param_stats(self):
        """Print network parameter statistics."""
        param_counts = self.get_param_count()
        
        print("Network Parameter Statistics:")
        print("-" * 50)
        
        # Print module-wise parameter counts
        for module_name, count in param_counts.items():
            if module_name != 'total':
                print(f"{module_name}: {count:,} parameters ({count * 4 / (1024**2):.2f} MB)")
        
        # Print total parameter count
        total = param_counts['total']
        print("-" * 50)
        print(f"Total parameters: {total:,} ({total * 4 / (1024**2):.2f} MB)")

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='meanflowql',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).

            # keep constant
            sigma=1.0,  # Noise scale
            consistency_alpha = 0.0,  # Consistency loss weight. 
            batch_size=256,  # Batch size.
            flow_ratio = 0.3,  # Control the rate of r==t, but we set r=0 in our experiments.  useless
            q_agg='mean',  # Aggregation method for target Q values.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            noise_type = "gaussian",  # The noise type, it can be uniform and gaussian.

            # critic config
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.

            # actor config 
            lr=1e-4,
            lr_min_ratio=0.1,
            actor_hidden_dims=256,  # Actor network hidden dimensions.
            actor_depth=3, # Transformer depth
            actor_num_heads = 2,# Transformer num heads  
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.  # hyper_paper 1 : keep same as fql                                              
            tau=0.005,  # Target network update rate. 
            alpha=10.0,  # BC coefficient (need to be tuned for each environment). # hyper_paper 2: most important
            tanh_squash=False,  # Whether to use tanh activation for the actor.
            use_output_layernorm=False,
            time_steps =  10000,  # Discrete time steps.  
            
            # meanflow loss
            adaptive_gamma=0.8,  # This parameter is used for controlling the loss function of meanflow.
            adaptive_c= 1e-4,     # Control the loss function of meanflow. 
            bound_loss_weight=1.0, # Control the bound_loss weight. 

            # best of N 
            num_candidates=5,  # Number of candidate actions for best-of-N selection. # hyper_paper 3 : 1 5
            action_mode="best", # action mode: control the way we sample actions.
            
            # Alpha scheduling configuration
            use_dynamic_alpha = True, # Boolean flag: True for dynamic, False for cosine
            alpha_update_interval=2000,  # Frequency of dynamic alpha adjustment
            loss_multiplier_threshold=5,  # High mean_flow_loss threshold for increasing alpha
            alpha_increase_factor=1.2,  # Factor to increase alpha when loss is high
            alpha_decrease_factor=0.8,  # Factor to decrease alpha when loss is low
            loss_history_window_size=20, # set the window size.
        )
    )
    return config
