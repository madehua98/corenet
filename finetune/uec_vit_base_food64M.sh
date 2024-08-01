export CFG_FILE=../projects/catlip/semantic_segmentation/uec/vit_base_food64M.yaml
export detection_results=/ML-A100/team/mm/models/catlip_data/results_vit_base/uec
CUDA_VISIBLE_DEVICES=0,1,2,3 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $detection_results --ddp.dist-url 'tcp://localhost:40002'
