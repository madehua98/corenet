export CFG_FILE=../projects/catlip/semantic_segmentation/vit_base.yaml
export detection_results=/ML-A100/team/mm/models/catlip_data/seg_vit_base
python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $detection_results
