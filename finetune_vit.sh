export CFG_FILE=projects/catlip/image_classification/imagenet/vit_base1.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/classification_results_vit
python corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file

