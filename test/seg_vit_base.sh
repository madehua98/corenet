export DATASET_PATH="/ML-A100/team/mm/models/FoodSeg103" # Change the path
export CFG_FILE=../projects/catlip/semantic_segmentation/vit_base.yaml
export MODEL_WEIGHTS=/ML-A100/team/mm/models/deeplabv3_vit_base.pt
CUDA_VISIBLE_DEVICES=0 python ../corenet/cli/main_eval.py  --common.config-file $CFG_FILE \
--model.segmentation.pretrained $MODEL_WEIGHTS \
--common.override-kwargs dataset.root_val=$DATASET_PATH
