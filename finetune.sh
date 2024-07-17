export CFG_FILE=projects/catlip/image_classification/food101/vitamin_small.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/single_small
python corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file
