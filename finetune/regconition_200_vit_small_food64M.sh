export CFG_FILE=../projects/catlip/image_classification/food200/vit_small_food64M.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/results_vit_small/9_food200
CUDA_VISIBLE_DEVICES=4,5,6,7 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.dist-url 'tcp://localhost:30006'
