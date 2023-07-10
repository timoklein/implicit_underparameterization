# Simplified benchmarking script from cleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/dqn.sh

OMP_NUM_THREADS=1 python -m src.benchmark \
    --env-ids PongNoFrameskip-v4 \
    --command "python feature_regularized_dqn.py --track True --regularize True" \
    --num-seeds 4 \
    --workers 2
