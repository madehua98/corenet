export CFG_FILE=../projects/catlip/linear_probing/food200/foodv_small_noc.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/results_small_noc/9_food200_lp
CUDA_VISIBLE_DEVICES=0,1,2,3 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.dist-url 'tcp://localhost:30011'
