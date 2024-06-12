CUDA_VISIBLE_DEVICES=0 corenet-train --common.config-file projects/catlip/pretraining/vit_base.yaml --common.results-loc results_catlip --ddp.rank 0 --ddp.world-size 1 --ddp.dist-url 'tcp://127.0.0.1:2345'

