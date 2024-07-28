export CFG_FILE=../projects/catlip/multi_label_image_classification/food172/foodv_small.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/results_small_dci/ingredient_172
CUDA_VISIBLE_DEVICES=4,5,6,7 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.dist-url 'tcp://localhost:30008'
