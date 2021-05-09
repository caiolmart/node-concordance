dglke_train \
--dataset ogbn-arxiv \
--data_path data/dataset_dglke \
--data_files entities.tsv relations.tsv \
ogbn-arxiv-edges-train.tsv ogbn-arxiv-edges-valid.tsv \
ogbn-arxiv-edges-test.tsv \
--format udd_hrt \
--model_name TransE_l2 --hidden_dim 500 \
--gamma 16.0 --lr 0.01 --batch_size 2000 \
--neg_sample_size 200 --max_step 5000 \
--log_interval 100 \
--batch_size_eval 16 --regularization_coef 1.00E-07 \
--gpu 0 \
--valid --test