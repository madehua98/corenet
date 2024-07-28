#!/bin/bash
export CONFIG_FILE=../projects/catlip/pretraining/foodv_base500.yaml
export RESULTS_FILE=/ML-A100/team/mm/models/catlip_data/results_base_dci
WORLD_SIZE=8 python ../corenet/cli/main_train.py --common.config-file $CONFIG_FILE --common.results-loc $RESULTS_FILE --ddp.rank 0 --ddp.world-size 8 --ddp.dist-url 'tcp://localhost:20000'
