from transformers import BertTokenizerFast
from transformers import GPT2Config,GPT2LMHeadModel
from torch.utils.data import DataLoader
import transformers
import argparse
import os
from os.path import join, exists
import torch
import pickle
from dataset import *
import torch.nn.utils.rnn as rnn_utils
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--model_config', default='config/config.json', type=str, required=False,
                        help='设置模型参数')
    parser.add_argument('--train_path', default='data/train.pkl', type=str, required=False, help='训练集路径')
    parser.add_argument('--max_len', default=150, type=int, required=False, help='训练时，输入数据的最大长度')

    parser.add_argument('--ignore_index', default=-100, type=int, required=False, help='对于ignore_index的label token不计算梯度')
    # parser.add_argument('--input_len', default=200, type=int, required=False, help='输入的长度')
    parser.add_argument('--epochs', default=100, type=int, required=False, help='训练的最大轮次')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='训练的batch size')
    parser.add_argument('--gpu0_bsz', default=10, type=int, required=False, help='0号卡的batch size')
    parser.add_argument('--lr', default=2.6e-5, type=float, required=False, help='学习率')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='衰减率')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, required=False)
    parser.add_argument('--save_model_path', default='model', type=str, required=False,
                        help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False,
                        help='预训练的模型的路径')
    # parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--patience', type=int, default=0, help="用于early stopping,设为0时,不进行early stopping.early stop得到的模型的生成效果不一定会更好。")
    parser.add_argument('--warmup_steps', type=int, default=4000, help='warm up步数')

    args = parser.parse_args()
    return args
def load_dataset(args):
    train_path = args.train_path

    with open(train_path, "rb") as f:
        input_list = pickle.load(f)
    train_dataset = MyDataset(input_list, args.max_len)
    return train_dataset

def collate_fen(batch):
    input_id = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch,batch_first=True,padding_value=-100) #忽略id 字典中倒数第100个，不是0，防止模型学到输入0输出0
    return input_id,labels
def train_epoch(model, train_dataloader,optimizer, scheduler,epoch, args):
    model.train()
    device = args.device
    ignore_index = args.ignore_index
    total_loss = 0  # 记录下整个epoch的loss的总和

    model_path = join(args.save_model_path, 'epoch{}'.format(epoch + 1))





    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        # 捕获cuda out of memory exception
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()
            total_loss += loss.item()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()
            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                print(str(exception))
                raise exception

    epoch_mean_loss = total_loss / len(train_dataloader)
    print("epoch：",epoch,"Loss:",epoch_mean_loss)
    model_path = join(args.save_model_path, 'epoch{}'.format(epoch + 1))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_path)

    return epoch_mean_loss



def train(model, train_dataset, args):
    data_load = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fen)
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    t_total = len(data_load) // args.gradient_accumulation_steps * args.epochs
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    train_losses, validate_losses = [], []
    for epoch in range(args.epochs):
        print("EPOCH:",epoch)
        # ========== train ========== #
        train_loss = train_epoch(
            model=model, train_dataloader=data_load,
            optimizer=optimizer, scheduler=scheduler,
           epoch=epoch, args=args)
        train_losses.append(train_loss)


def main():
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    args.cuda = not args.no_cuda
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda:0' if args.cuda else 'cpu'
    args.device = device

    tokenizer = BertTokenizerFast(vocab_file="vocab/vocab.txt", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    args.sep_id = tokenizer.sep_token_id
    args.pad_id = tokenizer.pad_token_id
    args.cls_id = tokenizer.cls_token_id

    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # 使用的是transformer的模型，没有预训练
    model_config = GPT2Config.from_json_file(args.model_config)
    model = GPT2LMHeadModel(config=model_config)
    model = model.to(device)
    assert model.config.vocab_size == tokenizer.vocab_size

    train_dataset = load_dataset(args)

    train(model, train_dataset, args)




if __name__ == '__main__':
    main()