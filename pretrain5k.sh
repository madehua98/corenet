#!/bin/bash
export CONFIG_FILE=projects/catlip/pretraining/vitamin_small500_1.yaml
export RESULTS_FILE=/ML-A100/team/mm/models/catlip_data/results500_1
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 WORLD_SIZE=6 python corenet/cli/main_train.py --common.config-file $CONFIG_FILE --common.results-loc $RESULTS_FILE --ddp.rank 0 --ddp.world-size 6 --ddp.dist-url 'tcp://localhost:40002'
#CUDA_VISIBLE_DEVICES=4,5,6 WORLD_SIZE=3 python corenet/cli/main_train.py --common.config-file $CONFIG_FILE --common.results-loc $RESULTS_FILE --ddp.rank 0 --ddp.world-size 3 --ddp.dist-url 'tcp://localhost:40000'
