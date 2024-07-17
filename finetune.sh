export CFG_FILE=projects/catlip/image_classification/imagenet/vitamin_base.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/single_base_500
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WORLD_SIZE=8 python corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.world-size 8
