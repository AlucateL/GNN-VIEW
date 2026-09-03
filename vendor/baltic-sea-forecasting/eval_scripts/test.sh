#!/bin/sh

MODEL=graph_efm
GRAPH=hierarchical

# Test training
python -m neural_lam.train_model\
    --config_path tests/datastore_examples/mdp/danra_100m_winds/config.yaml\
    --logger-project neural_lam_debug\
    --graph $GRAPH\
    --var_leads_val_plot '{"0": [1, 2], "1": [1, 2]}'\
    --val_steps_to_log 1 2\
    --hidden_dim 8\
    --processor_layers 2\
    --epochs 10\
    --val_interval 1\
    --model $MODEL\
    --ar_steps_train 2\
    --ar_steps_eval 4\
    --n_example_pred 1\
    --kl_beta 1\
    --crps_weight 1.\
    --eval test \
