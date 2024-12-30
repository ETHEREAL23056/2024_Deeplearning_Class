# 2024 Deep Learning Class Final Work Codes

Dataset from https://www.kaggle.com/datasets/rmisra/news-category-dataset

## Clean

Get the cleaned dataset with training part, valid part and test part. The result will be stored in "./dataset". Change the arguments "cls_data"and "gen_data" to get data for classification task or generation task. Change "sample_rate" to adapt to size of dataset and select "show_charts" to control the presentation of statistic charts.

```cmd
python clean.py --cls_data --show_charts --sample_rate 0.1
```

## Train

Before training, set the right directory to save and load the trained models which is directly stored in "./model". Change the arguments "lr", "epoch", "head_number", "layer_number", "class_number" in train.py to adapt to your model structure. Change the arguments "train_cls" or "train_gen" to decide training classification model or generation model.

To train a Transformer model towards text classification on dataset, simply run:

```cmd
python train.py --train_cls
```

The basic parameters is learning rate: 2e-5, epoch: 100, head number: 1, layer number: 6, class number: 40.

Trained models will be stored with named "{type}\_model\_{lr}\_{epoch}\_{head number}\_{layer number}.pth"


## Test

If you want to evaluate the detection performance of a trained model, simply run:

```cmd
python eval.py --gen_model_2e-5_100_4_6.pth --test_gen --save_name prediction
```

The prediction result will be stored in "./predictions". **The prediction results of classification model must be ended with "\_cls" and generation model must be ended with "\_gen" eg. prediction_gen.** The prediction result for classification has three attributes: true, pred and probs while generation with only true and pred.

If you want to show a demo of attention weights, run ""show_attention" argument directly:

```cmd
python eval.py --gen_model_2e-5_100_4_6.pth --show_attention
```

