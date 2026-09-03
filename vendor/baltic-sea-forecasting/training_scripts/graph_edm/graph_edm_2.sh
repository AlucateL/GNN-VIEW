# Stage 2 of training a Graph EDM model on Baltic Sea data

# NOTE:
# Change the config_path to point to your dataset configuration file
# Change the --num_workers, --precision and --batch_size according to your system's capabilities
# Add the checkpoint path to the stage 1 model checkpoint

pdm run python -m neural_lam.train_model \
    --config_path /monolith/global_data/ml_datasets/baltic_sea/data/baltic_nl_config_small.yaml \
    --model EDM \
    --n_example_pred 1 \
    --precision bf16-mixed \
    --graph hierarchical \
    --num_workers 12 \
    --precision bf16-mixed \
    --hidden_dim 64 \
    --processor_layers 2 \
    --pred_residual \
    --num_past_forcing_steps 1 \
    --num_future_forcing_steps 1 \
    --ar_steps_train 1 \
    --batch_size 6 \
    --epochs 1000 \
    --lr 0.0001 \
    --ensemble_size 5 \
    --val_interval 30 \
    --ar_steps_eval 4 \
    --num_sanity_val_steps 0 \
    --val_steps_to_log 1 2 4 \
    --var_leads_val_plot '{"0":[1,4], "1":[1,4], "2":[1,4], "3":[1,4], "4":[1,4]}' \
    --devices 3 4 5 \
    --load saved_models/train-EDM-2x64-08_29_11-1835/last.ckpt
