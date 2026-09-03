
pdm run python -m neural_lam.train_model \
  --config_path /monolith/global_data/ml_datasets/baltic_sea/data/baltic_graph_rework.yaml \
  --model graph_fm \
  --graph hierarchical \
  --num_nodes 1 \
  --num_workers 2 \
  --epochs 200 \
  --batch_size 4 \
  --lr 0.001 \
  --hidden_dim 8 \
  --processor_layers 2 \
  --ar_steps_train 1 \
  --precision bf16-mixed \
  --ar_steps_eval 4 \
  --val_steps_to_log 1 2 4 \
  --devices 0 \
  --num_sanity_val_steps 0 \
  --var_leads_val_plot '{"10":[1,4], "7":[1,4], "1":[1,4]}'
