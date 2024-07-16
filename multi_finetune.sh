export CFG_FILE=projects/catlip/multi_label_image_classification/vitamin_small.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/multi_finetune
CUDA_VISIBLE_DEVICES=7 WORLD_SIZE=1 python corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.rank 0 --ddp.world-size 1 --ddp.dist-url 'tcp://localhost:40001'

