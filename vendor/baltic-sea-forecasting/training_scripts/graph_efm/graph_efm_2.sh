# Stage 2 of training a Graph-EFM model on Baltic Sea data
# 1-step ELBO training

# The goal of this training is to learn to capture the distribution at one step.
# Here all parts of the model is trained.
# For single time step prediction, we want the model here to learn to produce
# plausible fields both using samples from the variational and prior distribution.
# We also want to see some diversity in the samples when using the prior.
# If the model collapses to deterministic forecasts => reduce kl_beta
# If the model does not produce reasonable-looking samples from the prior
# => increase kl_beta
# However, the spread-skill-ratio is not crucial to be ~1 here, as that will be
# helped by the CRPS in the last step.

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
    --lr 0.001 \
    --kl_beta 1 \
    --crps_weight 0 \
    --ar_steps_train 1 \
    --epochs 400 \
    --val_interval 20 \
    --ar_steps_eval 4 \
    --num_sanity_val_steps 0 \
    --val_steps_to_log 1 2 4 \
    --var_leads_val_plot '{"0":[1,4], "1":[1,4], "2":[1,4], "3":[1,4], "4":[1,4]}' \
    --devices 1 2 \
    --load saved_models/train-graph_efm-2x64-08_29_10-7865/last.ckpt
