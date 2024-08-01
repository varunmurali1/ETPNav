
NODE_RANK=0
NUM_GPUS=1
outdir=pretrained/rxr_ce/mlm.sap_habitat_depth

# train
torchrun \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port=$1 \
    pretrain_src/pretrain_src/train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config pretrain_src/run_pt/r2r_model_config_dep.json \
    --config pretrain_src/run_pt/r2r_pretrain_habitat.json \
    --output_dir $outdir
