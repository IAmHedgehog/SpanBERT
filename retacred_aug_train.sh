CUDA_VISIBLE_DEVICES=0,1,2,3 python code/run_tacred.py \
  --do_train \
  --do_eval \
  --train_file datasets/retacred_aug/train.json \
  --dev_file datasets/retacred/dev.json \
  --test_file datasets/retacred/test.json \
  --model spanbert-base-cased \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --max_seq_length 128 \
  --output_dir tacred_dir_aug_train


CUDA_VISIBLE_DEVICES=0,1,2,3 python code/run_tacred.py \
  --do_eval \
  --eval_test \
  --train_file datasets/retacred_aug/train.json \
  --dev_file datasets/retacred/dev.json \
  --test_file datasets/retacred/test.json \
  --model spanbert-base-cased \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --max_seq_length 128 \
  --output_dir tacred_dir_aug_train