# Stage 1 of training a Graph CRPS model on Baltic Sea data

# NOTE:
# Change the config_path to point to your dataset configuration file
# Change the --num_workers, --precision and --batch_size according to your system's capabilities

# TO ADJUST: Constants below
BS=4
#CKPT= # Don't load checkpoint for step 1
PREC=bf16-mixed  # Change to bf16 if suitable

pdm run python -m neural_lam.train_model \
    --config_path /monolith/global_data/ml_datasets/baltic_sea/data/baltic_graph_rework.yaml \
    --model crps \
    --noise_embedding linear \
    --noise_dim 32 \
    --precision $PREC \
    --n_example_pred 1 \
    --graph cluster \
    --num_workers 4 \
    --hidden_dim 64 \
    --processor_layers 3 \
    --pred_residual \
    --vertical_propnets 1 \
    --num_past_forcing_steps 1 \
    --num_future_forcing_steps 1 \
    --ar_steps_train 1 \
    --batch_size $BS \
    --epochs 200 \
    --lr 0.001 \
    --ensemble_size 5 \
    --val_interval 5 \
    --ar_steps_eval 4 \
    --val_steps_to_log 1 4 \
    --var_leads_val_plot '{"0":[1,4], "1":[1,4], "2":[1,4], "3":[1,4], "4":[1,4], "5":[1,4], "6":[1,4], "7":[1,4]}'\
    --num_sanity_val_steps 0 \
    --devices 1 2 4
