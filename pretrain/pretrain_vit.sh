#!/bin/bash
export CONFIG_FILE=../projects/catlip/pretraining/vit_base500.yaml
export RESULTS_FILE=/ML-A100/team/mm/models/catlip_data/results_catlip5k_vit
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 WORLD_SIZE=7 python ../corenet/cli/main_train.py --common.config-file $CONFIG_FILE --common.results-loc $RESULTS_FILE --ddp.rank 0 --ddp.world-size 7 --ddp.dist-url 'tcp://localhost:40000'
#CUDA_VISIBLE_DEVICES=4,5,6 WORLD_SIZE=3 python corenet/cli/main_train.py --common.config-file $CONFIG_FILE --common.results-loc $RESULTS_FILE --ddp.rank 0 --ddp.world-size 3 --ddp.dist-url 'tcp://localhost:40000'
