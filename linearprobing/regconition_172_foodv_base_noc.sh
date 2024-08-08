export CFG_FILE=../projects/catlip/linear_probing/food172/foodv_base_noc.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/results_base_noc/19_food172_lp
CUDA_VISIBLE_DEVICES=0,1,2,3 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.dist-url 'tcp://localhost:40002'
