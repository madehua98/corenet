export CFG_FILE=../projects/catlip/image_classification/food172/vit_small_food64M.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/results_vit_small/9_food172
CUDA_VISIBLE_DEVICES=0,1,2,3 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.dist-url 'tcp://localhost:30011'
