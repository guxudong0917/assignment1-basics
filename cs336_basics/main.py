from cs336_basics.Transformer import Transformer_lm,softmax
from cs336_basics.tokenizer import Tokenizer
import numpy as np
import torch,os
from cs336_basics.transformer_train import train_script
import time

import torch,wandb,random

def prepare_data(train_data_path,val_data_path,train_token_path,val_token_path):
    #初始化tokenizer
    vocab_path="/root/autodl-tmp/cs336_basics/tokenizer_data/vocab_tinystory.pkl"
    merges_path="/root/autodl-tmp/cs336_basics/tokenizer_data/merges_tinystory.pkl"
    special_tokens=["<|endoftext|>"]

    tokenizer=Tokenizer.from_files(vocab_filepath=vocab_path,merges_filepath=merges_path,special_tokens=special_tokens)

    #将txt文本encode,转为存入token ID的二进制文件
    with open(train_data_path,"r",encoding="utf-8") as train_data:
        #token 一个一个出来的,建立一个缓冲区,等到数字了就一起加进去
        buffer=[]
        with open(train_token_path,"wb") as f:
            for token in tokenizer.encode_iterable(train_data):
                buffer.append(token)
                if len(buffer) == 1000000:
                
                    f.write(np.array(buffer,dtype="uint16").tobytes())
                    buffer.clear()
            if len(buffer)>0:
                f.write(np.array(buffer,dtype="uint16").tobytes())


    with open(val_data_path,"r",encoding="utf-8") as val_data:
        #token 一个一个出来的,建立一个缓冲区,等到数字了就一起加进去
        buffer=[]
        with open(val_token_path,"wb") as f:
            for token in tokenizer.encode_iterable(val_data):
                buffer.append(token)
                if len(buffer) == 1000000:
                    
                    f.write(np.array(buffer,dtype="uint16").tobytes())
                    buffer.clear()
            if len(buffer)>0:
                f.write(np.array(buffer,dtype="uint16").tobytes())

name="batch64_owt_full_max_lr1e-3"

wandb.init(
    project="cs336-a1-train-model",          # 项目名（必选）
    name=name,        # 可读 run 名称
    config={
        "dataset_tag": "raw", # raw, sf, grpo
        "batch_size": 64,
        "max_iterations": "20000",
        "seed": 2026,
        "max_learning_rate":1e-3,
        "min_learning_rate": 1e-4,
        "warmup_iters":2000,
        "cosine_cycle_iters":20000,
        "betas":(0.9,0.95),
        "eps":1e-8,
        "weight_decay":0.1,
        "with_rope":True,
        "Norm_method":"preRnsNorm",
        "max_l2_norm":1
    }
)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 确保卷积操作等也使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    #选择设备
    device= "cuda" if torch.cuda.is_available() else "cpu"
    device=torch.device(device)

    #检查是否有转为token ID 的训练数据
    # train_token_path="/root/autodl-tmp/data/TinyStoriesV2-GPT4-train.bin"
    # val_token_path="/root/autodl-tmp/data/TinyStoriesV2-GPT4-valid.bin"

    train_token_path="/root/autodl-tmp/data/owt-train.bin"
    val_token_path="/root/autodl-tmp/data/owt-valid.bin"

    if os.path.exists(train_token_path) == False or os.path.exists(val_token_path) ==False:
        train_data_path="/root/autodl-tmp/data/TinyStoriesV2-GPT4-train.txt"
        val_data_path="/root/autodl-tmp/data/TinyStoriesV2-GPT4-valid.txt"

        # train_data_path="/root/autodl-tmp/data/owt_train.txt"
        # val_data_path="/root/autodl-tmp/data/owt_valid.txt"
        # prepare_data(train_data_path,val_data_path,train_token_path,val_token_path)

    
    save_place="result"
    save_main_path=os.path.join(save_place,name)

 
    set_seed(2026)
    model=Transformer_lm(vocab_size=32000,context_length=256,num_layers=4,d_model=512,num_heads=16,d_ff=1344,rope_theta=10000)

    model.to(device)
    
    train_log_file=os.path.join(save_main_path,"output/train_log")
    val_log_file=os.path.join(save_main_path,"output/val_log")

    os.makedirs(os.path.join(save_main_path,"output"),exist_ok=True)

    best_model_path=os.path.join(save_main_path,"checkpoint/best_model")
    save_model_path=os.path.join(save_main_path,"checkpoint/save_model")

    os.makedirs(best_model_path,exist_ok=True)
    os.makedirs(save_model_path,exist_ok=True)

    # summary_path=os.path.join(save_main_path,"summary_log")

    train_script(train_path=train_token_path,valid_path=val_token_path,
                 model=model,train_log_file=train_log_file,val_log_file=val_log_file,
                 best_model_path=best_model_path,save_model_path=save_model_path,
                 log_frequency=10,eval_frequency=500,checkpoint_frequency=4000,
                 max_learning_rate=1e-3,min_learning_rate=1e-4,warmup_iters=2000,cosine_cycle_iters=20000,
                 max_iterations=20000,eval_iterations=200,max_l2_norm=1,
                 context_length=256,device=device,batch_size=64,
                 lr=0.001,betas=(0.9,0.95),eps=1e-8,weight_decay=0.1
                 )


if __name__ == '__main__':
    main()