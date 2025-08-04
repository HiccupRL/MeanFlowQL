from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=1,
    train_dataset=None,  # 添加训练数据集参数，用于获取归一化统计信息
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        train_dataset: Training dataset with normalization statistics.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    # 检查是否需要进行归一化
    normalize_obs = False
    if train_dataset is not None and hasattr(train_dataset, 'normalize_obs') and train_dataset.normalize_obs:
        normalize_obs = True
        obs_mean = train_dataset.obs_mean
        obs_std = train_dataset.obs_std
        print("Using observation normalization during evaluation")

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        while not done:
            # enable the observation have its own dimension. 
            obs_batch = np.expand_dims(observation, axis=0) if observation.ndim == 1 else observation
            
            if normalize_obs:
                obs_batch = (obs_batch - obs_mean) / obs_std
                
            action = actor_fn(observations=obs_batch, temperature=eval_temperature)
            
            action = np.array(action[0] if action.ndim > 1 and action.shape[0] == 1 else action)
            action = np.clip(action, -1, 1)
            next_observation, reward, terminated, truncated, info = env.step(action)
            if i>=num_eval_episodes-1:
                # print(f"current observation is {observation}")
                print(f"Step {i}: The agent generated action is {action}")
                # print(f"next observation is {next_observation}")
                # print(f"reward is {reward}")
            
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
