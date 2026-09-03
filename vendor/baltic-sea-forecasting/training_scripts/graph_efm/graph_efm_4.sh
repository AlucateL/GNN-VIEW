# Stage 4 of training a Graph-EFM model on Baltic Sea data
# Unrolled training ELBO + CRPS

# This is the final fine-tuning, with the added CRPS loss.
# This is really the first time the latent map is trained in a roll-out setting.
# It should help both with calibrating the spread (closer to
# spread-skill-ratio = 1) and with reducing RMSE over longer lead times.
# If no improvement is seen for the things mentioned above
# => increase --crps_weight (by a factor x10)
# If spatially incoherent artefacts start appearing => decrease --crps_weight

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
    --crps_weight 1e6 \
    --ar_steps_train 4 \
    --epochs 700 \
    --val_interval 20 \
    --ar_steps_eval 10 \
    --num_sanity_val_steps 0 \
    --val_steps_to_log 1 2 4 10 \
    --var_leads_val_plot '{"0":[1,4], "1":[1,4], "2":[1,4], "3":[1,4], "4":[1,4]}' \
    --devices 5 \
    --load saved_models/train-graph_efm-2x64-09_03_10-4068/last.ckpt
