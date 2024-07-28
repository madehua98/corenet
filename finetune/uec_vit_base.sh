export CFG_FILE=../projects/catlip/semantic_segmentation/uec/vit_base.yaml
export detection_results=/ML-A100/team/mm/models/catlip_data/catlip_vit_base/uec_224
CUDA_VISIBLE_DEVICES=4,5,6,7 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $detection_results --ddp.dist-url 'tcp://localhost:4005'
