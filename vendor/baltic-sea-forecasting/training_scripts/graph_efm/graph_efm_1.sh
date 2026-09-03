# Stage 1 of training a Graph-EFM model on Baltic Sea data
# "Auto-encoder" training

# The goal of this training is only for the model to learn to encode information
# into the latent variable, and decode this to feasible fields.
# The latent map (prior) is not trained here.
# We want the vi samples (predicted using variational distribution) to be as close
# to the trues states as possible.
# Since the prior isn't trained, the prior samples do not matter here.

pdm run python -m neural_lam.train_model \
    --config_path /monolith/global_data/ml_datasets/baltic_sea/data/baltic_graph_rework.yaml \
    --num_workers 2 \
    --precision bf16-mixed \
    --model graph_efm \
    --graph cluster \
    --hidden_dim 128 \
    --processor_layers 6 \
    --num_past_forcing_steps 1 \
    --num_future_forcing_steps 1 \
    --n_example_pred 1 \
    --batch_size 1 \
    --lr 0.001 \
    --kl_beta 0 \
    --crps_weight 0 \
    --ar_steps_train 1 \
    --epochs 200 \
    --val_interval 5 \
    --ar_steps_eval 4 \
    --num_sanity_val_steps 0 \
    --val_steps_to_log 1 4 \
    --var_leads_val_plot '{"0":[1,4], "1":[1,4], "2":[1,4], "3":[1,4], "4":[1,4]}' \
    --devices 5
