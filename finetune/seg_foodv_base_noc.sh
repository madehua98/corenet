export CFG_FILE=../projects/catlip/semantic_segmentation/seg/foodv_base_noc.yaml
export detection_results=/ML-A100/team/mm/models/catlip_data/results_base_noc/19_seg_224
CUDA_VISIBLE_DEVICES=0,1,2,3 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $detection_results --ddp.dist-url 'tcp://localhost:40011'
