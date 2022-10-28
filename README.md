# Chinese Text Classification

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

中文文本分类，TextCNN, TextRNN, Transformer, 基于pytorch，开箱即用。

## 介绍

本项目是用于NLP深度学习入门或文本分类快速生产使用。

> 项目模型代码引用自[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)

## 环境
python 3.7  
pytorch 1.1  
tqdm

## 中文数据集
本项目从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万


### 更换自己的数据集

可以自行制作一份csv文件, 确保第一列为文本, 第二列为类别标签即可


## 效果

模型|acc|备注
--|--|--
TextCNN|91.22%|Kim 2014 经典的CNN文本分类
TextRNN|91.12%|BiLSTM 
Transformer|89.91%|效果较差

## 使用说明

在游乐场`playground.ipynb`中运行项目代码, 快速上手学习**Pytorch**.

### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。  


## 对应论文
[1] Convolutional Neural Networks for Sentence Classification  
[2] Recurrent Neural Network for Text Classification with Multi-Task Learning  
[3] Attention Is All You Need  
