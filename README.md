# One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow

## Setup

### Environment Setup

```bash
conda env create -f environment.yml
conda activate flowrl
```

### Dataset Setup

```
python download_all_datasets.py
```

## Toy Experiments

```
python toy_example/verify_fql_flow_fit.py
python toy_example/verify_meanflow_uat.py
```

If you wish to evaluate the performance of our different variants on the toy example, please modify the training objective in the `meanflow_loss` section and the action sampling logic in the `sample_action` function within the `meanflowql_beta.py` file accordingly.

## Offline Experiments

```
Example:
python main_meanflowql.py --env_name=humanoidmaze-large-navigate-singletask-task1-v0 --agent=agents/meanflowql.py --agent.time_steps=50 --early_stopping_metric=evaluation/success --offline_steps=1000005 --early_stopping_patience=10 --proj_wandb=0726_humanoidmaze-large-navigate-singletask-task1-v0 --wandb_save_dir=07281632_meanflowRL_param_search --run_group=meanflow_param_search --seed=1 --agent.alpha=6000 --agent.discount=0.995 --agent.num_candidates=5 --wandb_online=True
```

## Offline to Online Experiments

```
Example
python main_meanflowql.py --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --agent=agents/meanflowql.py --online_steps=1000000 --offline_steps=1000000 --proj_wandb=0729online-humanoidmaze-medium-navigate-singletask-task1-v0 --wandb_save_dir=meanflowRL_online --run_group=meanflow_online --seed=1 --early_stopping_patience=100 --wandb_online=True --agent.time_steps=100 --agent.alpha=2000 --agent.discount=0.995 --agent.num_candidates=5
```
