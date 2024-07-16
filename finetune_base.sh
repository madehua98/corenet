export CFG_FILE=projects/catlip/image_classification/imagenet/vitamin_base.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/single_base_500
python corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file
