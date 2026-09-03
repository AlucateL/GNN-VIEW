# Stage 3 of training a Graph-EFM model on Baltic Sea data
# Unrolled ELBO training

# This is the same as the previous step, but unrolled over multiple steps.
# The same instructions apply, but keep in particular in mind that we want to
# see some diversity in prior samples as they are rolled out for longer time.

pdm run python -m neural_lam.train_model \
    --config_path /monolith/global_data/ml_datasets/baltic_sea/data/baltic_nl_config_small.yaml \
    --num_workers 8 \
    --precision bf16-mixed \
    --model graph_efm \
    --graph hierarchical \
    --hidden_dim 64 \
    --processor_layers 2 \
    --num_past_forcing_steps 1 \
    --num_future_forcing_steps 1 \
    --n_example_pred 1 \
    --ensemble_size 5 \
    --batch_size 2 \
    --lr 0.0001 \
    --kl_beta 1 \
    --crps_weight 0 \
    --ar_steps_train 4 \
    --epochs 600 \
    --val_interval 20 \
    --ar_steps_eval 10 \
    --num_sanity_val_steps 0 \
    --val_steps_to_log 1 2 4 10 \
    --var_leads_val_plot '{"0":[1,4], "1":[1,4], "2":[1,4], "3":[1,4], "4":[1,4]}' \
    --devices 4 \
    --load saved_models/train-graph_efm-2x64-09_02_12-5629/last.ckpt
