# README

This is the implementation of the ICLR 4064 submission "COPA: Certifying Robust Policies for Offline Reinforcement Learning against Poisoning Attacks". The code is adapted on the basis of the offline RL training repo https://github.com/google-research/batch_rl.

Basically, we provide two certification (**per-state action certification** and **reward certification**) for three aggregation protocols (**PARL, TPARL, DPARL**). Below we present the example commands for running these certifications.

## Dataset Partitioning via Hashing

1. Generate the trajectory indices for each hash num in $[50]$: 

```bash
python split.py --train-data-folder /data/common/kahlua/dqn_replay/$1/$2/replay_logs \
                --output-folder /data/common/kahlua/dqn_replay/hash_split/$1_$2
```

With the above command in ``split_script.sh``, simply run the following commands, e.g., for the game Pong.

```bash
bash split_script.sh Pong 1
bash split_script.sh Pong 2
bash split_script.sh Pong 3
bash split_script.sh Pong 4
bash split_script.sh Pong 5
```

2. For each hash number, generate the corresponding datasets

```bash
python gen_split.py --train-data-folder /data/common/kahlua/dqn_replay/$3/$1/replay_logs \
                    --epi-index-path /data/common/kahlua/dqn_replay/hash_split/$3_$1/partition_$2.pt \
                    --output-folder /data/common/kahlua/dqn_replay/hash_split/$3_$1/dataset/hash_$2 \
                    --start-id 0 --end-id 50
```

With the above command in ``gen_split_script.sh``, simply run the following commands, e.g., for the game Pong.

```bash
bash gen_split_script.sh 1 0 Pong
bash gen_split_script.sh 2 0 Pong
bash gen_split_script.sh 3 0 Pong
bash gen_split_script.sh 4 0 Pong
bash gen_split_script.sh 5 0 Pong
```

The above commands generate the $5$ datasets for hash number $0$. We would repeat the above commands for $50$ times to generate the datasets for hash number $0\sim49$.

3. For each hash number, merge the $5$ Datasets

```bash
python merge_splits.py --input-folder /data/common/kahlua/dqn_replay/hash_split/Pong --hash-num 0
```

The above command merges the $5$ Datasets for hash number $0$. Repeat it for $50$ times for all hash numbers.

## Model Training

The following command trains the model based on the datasets Pong of hash number $1$ using RL algorithm DQN for $100$ iterations.

```bash
CUDA_VISIBLE_DEVICES=2 python -um batch_rl.fixed_replay.train   \
		--base_dir=/data/common/kahlua/COPA/Pong/hash_1  \
    --replay_dir=/data/common/kahlua/dqn_replay/hash_split/Pong/hash_1/ \
    --gin_files='batch_rl/fixed_replay/configs/dqn.gin' \
    --gin_bindings='atari_lib.create_atari_environment.game_name = "Pong"' \
    --gin_bindings='FixedReplayRunner.num_iterations=100'
```

## Certifying Per-State Action

1. PARL

```bash
python -um batch_rl.fixed_replay.test \
			 --base_dir [base_dir] --model_dir [model_dir] \
			 --cert_alg tight \
			 --total_num 50 --max_steps_per_episode 1000 \
			 --agent_name dqn \                                                                                                                             			 --gin_files='copa/fixed_replay/configs/dqn.gin' \                                                                                                                                 			  --gin_bindings='atari_lib.create_atari_environment.game_name = "Freeway"'
```

where `base_dir` is the path for storing experimental logs and results, and `model_dir` is the path of trained $u$ subpolicies.

2. TPARL

```bash
python -um batch_rl.fixed_replay.test \
			 --base_dir [base_dir] --model_dir [model_dir] \
			 --cert_alg window --window_size 4 \
			 --total_num 50 --max_steps_per_episode 1000 \
			 --agent_name dqn \                                                                                                                             			 --gin_files='copa/fixed_replay/configs/dqn.gin' \                                                                                                                                 			  --gin_bindings='atari_lib.create_atari_environment.game_name = "Freeway"'
```

For TPARL, we explicitly pass the `cert_alg` option as `window` and configure the predetermined window size $W$.

3. DPARL

```bash
python -um batch_rl.fixed_replay.test \
			 --base_dir [base_dir] --model_dir [model_dir] \
			 --cert_alg dynamic --max_window_size 5 \
			 --total_num 50 --max_steps_per_episode 1000 \
			 --agent_name dqn \                                                                                                                             			 --gin_files='copa/fixed_replay/configs/dqn.gin' \                                                                                                                                 			  --gin_bindings='atari_lib.create_atari_environment.game_name = "Freeway"'
```

For DPARL, we explicitly pass the `cert_alg` option as `dynamic` and configure the maximum window size $W_{\rm max}$.

## Certifying Cumulative Reward

1. PARL

```bash
python -um batch_rl.fixed_replay.test_reward \
			 --base_dir [base_dir] --model_dir [model_dir] \
			 --cert_alg tight \
			 --total_num 50 --max_steps_per_episode 1000 \
			 --agent_name dqn \                                                                                                                             			 --gin_files='copa/fixed_replay/configs/dqn.gin' \                                                                                                                                 			  --gin_bindings='atari_lib.create_atari_environment.game_name = "Freeway"'
```

where `base_dir` is the path for storing experimental logs and results, and `model_dir` is the path of trained $u$ subpolicies.

2. TPARL

```bash
python -um batch_rl.fixed_replay.test_reward \
			 --base_dir [base_dir] --model_dir [model_dir] \
			 --cert_alg window --window_size 4 \
			 --total_num 50 --max_steps_per_episode 1000 \
			 --agent_name dqn \                                                                                                                             			 --gin_files='copa/fixed_replay/configs/dqn.gin' \                                                                                                                                 			  --gin_bindings='atari_lib.create_atari_environment.game_name = "Freeway"'
```

For TPARL, we explicitly pass the `cert_alg` option as `window` and configure the predetermined window size $W$.

3. DPARL

```bash
python -um batch_rl.fixed_replay.test_reward \
			 --base_dir [base_dir] --model_dir [model_dir] \
			 --cert_alg dynamic --max_window_size 5 \
			 --total_num 50 --max_steps_per_episode 1000 \
			 --agent_name dqn \                                                                                                                             			 --gin_files='copa/fixed_replay/configs/dqn.gin' \                                                                                                                                 			  --gin_bindings='atari_lib.create_atari_environment.game_name = "Freeway"'
```

For DPARL, we explicitly pass the `cert_alg` option as `dynamic` and configure the maximum window size $W_{\rm max}$.
