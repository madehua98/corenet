export CFG_FILE=../projects/catlip/multi_label_image_classification/food200/foodv_base_noc.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/results_base_noc/19_ingredient_200
CUDA_VISIBLE_DEVICES=4,5,6,7 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.dist-url 'tcp://localhost:30013'
