# GPT2多轮对话闲聊机器人

#### 介绍
使用GPT2实现多轮对话闲聊机器人

#### 软件架构
使用GPT2训练多轮对话聊天机器人

改编自：https://github.com/yangjianxin1/GPT2-chitchat

pythorch 版本

#### 使用说明

训练集：链接：https://pan.baidu.com/s/1q0_3wex2-FeKfnBXEmsgVg 提取码：awe5 

将训练集下载后改名train.txt 放到data文件夹下

训练完成的模型:链接：https://pan.baidu.com/s/1aN9k6oY-Da1d4cDIORbZQA 提取码：e56r 
将epoch40 放入model文件夹



1.  python preprocess.py 对数据集预处理
2.  python train.py 训练
3.  python main.py 预测
