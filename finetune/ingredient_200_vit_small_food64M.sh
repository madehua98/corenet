export CFG_FILE=../projects/catlip/multi_label_image_classification/food200/vit_small_food64M.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/results_vit_small/9_ingredient_200
CUDA_VISIBLE_DEVICES=0,1,2,3 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.dist-url 'tcp://localhost:30111'
