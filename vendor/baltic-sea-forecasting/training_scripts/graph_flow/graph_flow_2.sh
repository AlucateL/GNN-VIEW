# Stage 2 of training a Graph flow matching model on Baltic Sea data

# NOTE:
# Change the config_path to point to your dataset configuration file
# Change the --num_workers, --precision and --batch_size according to your system's capabilities
# Add the checkpoint path to the stage 1 model checkpoint

# TO ADJUST: Constants below
BS=1
#CKPT= # Don't load checkpoint for step 1
PREC=32 # Change to bf16 if suitable

python -m neural_lam.train_model \
    --config_path data/baltic_nl_config_small.yaml \
    --model FM \
    --n_example_pred 1 \
    --precision $PREC\
    --graph hierarchical \
    --num_workers 16 \
    --precision 32 \
    --hidden_dim 128 \
    --processor_layers 1 \
    --pred_residual \
    --vertical_propnets 1 \
    --num_past_forcing_steps 1 \
    --num_future_forcing_steps 1 \
    --ar_steps_train 1\
    --batch_size $BS \
    --epochs 1000 \
    --lr 0.0001 \
    --ensemble_size 5\
    --val_interval 30\
    --ar_steps_eval 4\
    --val_steps_to_log 1 2 4\
    --var_leads_val_plot '{"10":[1,4], "7":[1,4], "1":[1,4]}'\
    --load checkpoint_path
