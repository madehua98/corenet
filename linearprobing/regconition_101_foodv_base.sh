export CFG_FILE=../projects/catlip/linear_probing/food101/foodv_base.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/results_base_dci/19_food101_lp
CUDA_VISIBLE_DEVICES=4,5,6,7 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.dist-url 'tcp://localhost:30006'
