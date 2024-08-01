export CFG_FILE=../projects/catlip/image_classification/food200/openvit_base.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/open_vit_base/food200
CUDA_VISIBLE_DEVICES=1,2,3,0 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.dist-url 'tcp://localhost:30003'
