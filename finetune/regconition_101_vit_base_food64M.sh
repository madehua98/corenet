export CFG_FILE=../projects/catlip/image_classification/food101/vit_base_food64M.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/results_vit_base/9_food101
CUDA_VISIBLE_DEVICES=4,5,6,7 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.dist-url 'tcp://localhost:30006'
