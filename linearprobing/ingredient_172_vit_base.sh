export CFG_FILE=../projects/catlip/linear_probing/food172/vit_base.yaml
export result_file=/ML-A100/team/mm/models/catlip_data/ingredient172_lp
CUDA_VISIBLE_DEVICES=0,1,2,3 python ../corenet/cli/main_train.py --common.config-file $CFG_FILE --common.results-loc $result_file --ddp.dist-url 'tcp://localhost:30001'
