export CFG_FILE=projects/catlip/image_classification/imagenet/vitamin_small1.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/classification_results1
python corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file
