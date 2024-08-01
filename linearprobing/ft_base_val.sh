export CFG_FILE=projects/catlip/image_classification/food101/vitamin_base.yaml
export DATASET_PATH="/ML-A100/team/mm/models/food101/food101/test_images" # change to the ImageNet validation path
export MODEL_WEIGHTS=/ML-A100/team/mm/models/catlip_data/single_base_500/train/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=1 python corenet/cli/main_eval.py --common.config-file $CFG_FILE --common.override-kwargs dataset.root_val=$DATASET_PATH model.classification.pretrained=$MODEL_WEIGHTS model.resume_exclude_scopes=''
