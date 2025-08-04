import os
import platform
# Set OpenGL platform to EGL for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# Configure MuJoCo to use EGL renderer
os.environ['MUJOCO_GL'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# # Limit GPU memory usage to 50%
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
import json
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb


FLAGS = flags.FLAGS

# Experiment configuration flags
flags.DEFINE_string('run_group', 'debug', 'Run group in wandb')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-umaze-v2', 'Environment (dataset) name.')
flags.DEFINE_string('proj_wandb', 'flow_RL', 'wandb project name')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('wandb_save_dir', 'debug/', 'Wandb offline data save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_boolean('use_observation_normalization', True, 'Whether to normalize observations')

# Training configuration flags
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

# Evaluation configuration flags
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.') 

# Data augmentation and preprocessing flags
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_float('pretrain_factor', 0.0, 'Fraction of offline steps used for pretraining')
flags.DEFINE_boolean('wandb_online', True , 'Whether to use wandb online mode')

# Early stopping co nfiguration flags
flags.DEFINE_bool('enable_early_stopping', True, 'Whether to enable early stopping.')
flags.DEFINE_integer('early_stopping_patience', 6, 'Number of evaluations to wait before early stopping.')
flags.DEFINE_float('early_stopping_min_delta', 0.01, 'Minimum change in validation metric to qualify as improvement.')
flags.DEFINE_string('early_stopping_metric', 'evaluation/success', 'Metric to monitor for early stopping.')

config_flags.DEFINE_config_file('agent', 'agents/meanfql_dit_inv_simp_dyna_alpha.py', lock_config=False)


def main(_):
    """
    Main training function for MeanFlow RL with early stopping.
    
    This code is modified based on the original main.py.
    The experiment hyperparameters remain the same, with modifications
    to input/output interfaces and addition of early stopping functionality.
    """
    # Initialize experiment logging
    exp_name = get_exp_name(FLAGS.seed)
    import os
    # Configure wandb mode based on command line arguments
    if FLAGS.wandb_online:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "offline"
    setup_wandb(project=FLAGS.proj_wandb, group=FLAGS.run_group, name=exp_name, mode= os.environ["WANDB_MODE"],wandb_output_dir = FLAGS.wandb_save_dir)
    

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    print(f"Saving results to {FLAGS.save_dir}") 
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Initialize environment and datasets
    config = FLAGS.agent
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
    if FLAGS.online_steps > 0:
        assert 'visual' not in FLAGS.env_name, 'Online fine-tuning is currently not supported for visual environments.'

    # Set random seeds for reproducibility
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Configure datasets and replay buffer
    train_dataset = Dataset.create(**train_dataset)
    if FLAGS.balanced_sampling:
        # Create separate replay buffer for balanced sampling between training dataset and replay buffer
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        # Use training dataset as the replay buffer
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    # Configure data augmentation and frame stacking for all datasets
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] == 'rebrac':
                dataset.return_next_actions = True
    
    if FLAGS.use_observation_normalization:
        print("Computing observation normalization statistics...")
        if train_dataset is not None:
            train_dataset.compute_normalization_stats()
            train_dataset.enable_normalization(True)
        if val_dataset is not None and train_dataset is not None:
            val_dataset.obs_mean = train_dataset.obs_mean
            val_dataset.obs_std = train_dataset.obs_std
            val_dataset.enable_normalization(True)
        if replay_buffer is not None and train_dataset is not None and replay_buffer != train_dataset:
            replay_buffer.obs_mean = train_dataset.obs_mean
            replay_buffer.obs_std = train_dataset.obs_std
            replay_buffer.enable_normalization(True)
    else:
        print("Observation normalization disabled")
    print("Observation normalization setup complete.")
    # Initialize agent
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    
    # Print agent parameter statistics
    agent.print_param_stats()

    # Restore agent from checkpoint if specified
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Initialize training loggers and timing
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    
    # Initialize online training variables

    
    # Initialize early stopping variables
    best_metric_value = float('-inf')  # For metrics where higher is better (like success)
    patience_counter = 0
    early_stopped = False
    
    # Determine if metric should be maximized or minimized
    maximize_metric = 'return' in FLAGS.early_stopping_metric or 'reward' in FLAGS.early_stopping_metric or 'success' in FLAGS.early_stopping_metric
    if not maximize_metric:
        best_metric_value = float('inf')  # For metrics where lower is better (like loss)

    # Pretraining phase
    pretrain_steps = int(FLAGS.offline_steps * FLAGS.pretrain_factor)
    for i in tqdm.tqdm(range(1, pretrain_steps + 1), smoothing=0.1, dynamic_ncols=True, desc="Pretrain"):
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.pretrain(batch, current_step=i)
        
        # Log pretraining metrics
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'pretraining/{k}': v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)
        
        # Evaluate agent during pretraining
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
                                       
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                train_dataset=train_dataset,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'pretrain_evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video
                
            wandb.log(eval_metrics, step=i)


    # Q-Learning phase
    q_learning_steps = FLAGS.offline_steps + FLAGS.online_steps
    
    # Reset early stopping variables for Q-Learning phase
    best_metric_value = float('-inf') if maximize_metric else float('inf')
    patience_counter = 0
    early_stopped = False
    
    for i in tqdm.tqdm(range(pretrain_steps, q_learning_steps + pretrain_steps+1), smoothing=0.1, dynamic_ncols=True, desc="Q-Learning"):
        if i <= FLAGS.offline_steps + pretrain_steps:
            # Offline reinforcement learning
            batch = train_dataset.sample(config['batch_size'])
    
            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(
                    batch, full_update=(i % config['actor_freq'] == 0)
                )
            else:
                agent, update_info = agent.update(
                    batch, current_step=i
                )
        else:
            # Online fine-tuning phase
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()
            
            # Ensure observation has correct shape before calling sample_actions
            if len(ob.shape) == 1:
                ob_batch = ob[None, :]  # Add batch dimension
            else:
                ob_batch = ob
            
            # Apply observation normalization if enabled
            if FLAGS.use_observation_normalization and train_dataset is not None and hasattr(train_dataset, 'normalize_obs') and train_dataset.normalize_obs:
                print("normalizing online")
                ob_batch = (ob_batch - train_dataset.obs_mean) / train_dataset.obs_std
            
            action = agent.sample_actions(observations=ob_batch, seed=key)
            action = np.array(action)
            
            # Remove batch dimension if present for environment step
            if action.ndim > 1 and action.shape[0] == 1:
                action = action[0]  # Remove batch dimension: (1, 8) -> (8,)
            

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            # Apply D4RL antmaze reward adjustment
            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0

            # Store transition in replay buffer (use original unnormalized observations)
            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent with appropriate sampling strategy
            if FLAGS.balanced_sampling:
                # Sample half from training dataset and half from replay buffer
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
            else:
                batch = train_dataset.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(
                    batch, full_update=(i % config['actor_freq'] == 0),
                    current_step=i
                )
            else:
                agent, update_info = agent.update(
                    batch, current_step=i
                )

        # Log training metrics
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                # Compute validation loss with required rng parameter
                _, val_info = agent.total_loss(val_batch, grad_params=None, rng=agent.rng, current_step=i)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent performance
        if FLAGS.eval_interval != 0 and (i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                train_dataset=train_dataset,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video
                
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)
            
            # Enhanced early stopping check for Q-Learning phase
            if FLAGS.enable_early_stopping and FLAGS.early_stopping_metric in eval_metrics:
                current_metric = eval_metrics[FLAGS.early_stopping_metric]
                
                # Special handling: trigger early stopping if both success and return remain at 0
                success_rate = eval_metrics.get('evaluation/success', 0.0)
                return_value = eval_metrics.get('evaluation/return', 0.0)
                
                # Check for complete lack of learning progress (both success and return at 0)
                if success_rate == 0.0 and return_value == 0.0:
                    patience_counter += 1
                    print(f"Q-Learning: No learning progress (success=0, return=0). Patience: {patience_counter}/{FLAGS.early_stopping_patience}")
                    
                    if patience_counter >= FLAGS.early_stopping_patience:
                        print(f"Early stopping triggered due to no learning progress at step {i}")
                        early_stopped = True
                        break
                else:
                    # Standard early stopping logic based on metric improvement
                    if maximize_metric:
                        improved = current_metric > best_metric_value + FLAGS.early_stopping_min_delta
                    else:
                        improved = current_metric < best_metric_value - FLAGS.early_stopping_min_delta
                    
                    if improved:
                        best_metric_value = current_metric
                        patience_counter = 0
                        print(f"Q-Learning: New best {FLAGS.early_stopping_metric}: {current_metric:.4f}")
                    else:
                        patience_counter += 1
                        print(f"Q-Learning: No improvement in {FLAGS.early_stopping_metric}. Patience: {patience_counter}/{FLAGS.early_stopping_patience}")
                    
                    if patience_counter >= FLAGS.early_stopping_patience:
                        print(f"Early stopping triggered during Q-Learning at step {i}")
                        early_stopped = True
                        break

        # Save agent checkpoint
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)
            
        # Exit training loop if early stopping was triggered
        if early_stopped:
            print(f"Q-Learning: Early stopping triggered at step {i}")
            wandb.log({
                'early_stopping/step': i,
                'early_stopping/metric': FLAGS.early_stopping_metric,
                'early_stopping/best_metric_value': best_metric_value,
            })
            train_logger.log(
                {
                    'early_stopping/step': i,
                    'early_stopping/metric': FLAGS.early_stopping_metric,
                    'early_stopping/best_metric_value': best_metric_value,
                }
            )
            break

    # Cleanup and finalize logging
    wandb.finish()
    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
