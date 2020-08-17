# train
+ run script
```
export PYTHONPATH=`pwd`:${PYTHONPATH:-}
python run/main.py --train_mode train --input data/BaiduIE_2020/ --output finetune_model_path/ --bert_model hfl/chinese-roberta-wwm-ext --max_len 128 --train_batch_size 64 --learning_rate 2e-5 --epoch_num 20 --patience_stop 13 
```
  
