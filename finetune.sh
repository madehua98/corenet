export CFG_FILE=projects/catlip/image_classification/imagenet/vitamin_small1.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/single_small_500_dci
python corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file
