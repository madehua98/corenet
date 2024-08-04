export CFG_FILE=/ML-A800/home/guoshuyue/madehua/code/corenet/projects/catlip/image_classification/food101/foodv_small.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/results_small_dci/9_food101
CUDA_VISIBLE_DEVICES=4,5,6,7 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.dist-url 'tcp://localhost:30006'
