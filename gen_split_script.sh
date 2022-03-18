set -x
python gen_split.py --train-data-folder /data/common/kahlua/dqn_replay/$3/$1/replay_logs \
	                    --epi-index-path /data/common/kahlua/dqn_replay/hash_split/$3_$1/partition_$2.pt \
			                        --output-folder /data/common/kahlua/dqn_replay/hash_split/$3_$1/dataset/hash_$2 \
						                    --start-id 0 --end-id 50

