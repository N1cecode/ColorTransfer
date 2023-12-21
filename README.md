# Commands

## Dataset

### Soft link dataset to repo

~~~bash
ln -s /path/to/dataset/ ./data/
~~~

## Train

~~~bash
python main.py --mode train --num_domains 1 --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 --batch_size 1 --val_batch_size 8 --lr 2e-4 --train_img_dir /path/to/train/dataset/ --val_img_dir /path/to/val/dataset/
~~~
