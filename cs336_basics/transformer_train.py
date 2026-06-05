import torch.nn as nn
import torch
import einops
import math
import numpy.typing as npt
from jaxtyping import Bool, Float, Int
from torch import Tensor

from collections.abc import Callable, Iterable
from typing import Optional

from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
import os,typing
import time
import json
from cs336_basics.Transformer import Transformer_lm,softmax

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):

    max_logits=torch.max(inputs,dim=-1,keepdim=True).values

    logits=inputs-max_logits
    batch_size=targets.shape[0]

    loss =(-logits[torch.arange(batch_size,device=inputs.device),targets]+torch.log(torch.sum(torch.exp(logits),dim=-1)))
    #loss得到的是一个向量，关于每一个位置的预测损失,要求平均
    return loss.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}

        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:

            """
                每个param_group包括模型的可训练参数,和defaults部分设置的学习率等等
                eg:
                ---
                 optimizer = AdamW(
                    [
                        {"params": decay_params, "weight_decay": 0.1},
                        {"params": no_decay_params, "weight_decay": 0.0},
                    ],
                    lr=1e-3,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )

                这里：

                - 两组都有默认的 lr=1e-3
                - 两组都有默认的 betas
                - 两组都有默认的 eps
                - weight_decay 被每组单独覆盖

                最终每个 param_group 都会包含完整配置，不会缺少 eps

                group 0: lr, betas, eps, weight_decay=0.1, params
                group 1: lr, betas, eps, weight_decay=0.0, params
            """

            lr = group["lr"] # 这里它读取的是对应
            beta1=group["betas"][0]
            beta2=group["betas"][1]

            eps=group["eps"]
            weight_decay=group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.

                grad = p.grad.data # Get the gradient of loss with respect to p.
                
                if len(state)==0:
                    state["m"]=torch.zeros_like(grad)
                    state["v"]=torch.zeros_like(grad)

                t = state.get("t", 0)+1 # Get iteration number from the state, or 0.

                lr_t=lr*math.sqrt(1-pow(beta2,t))/(1-pow(beta1,t))

                #权重衰减,用原始学习率
                p.data-=lr*weight_decay*p.data

               

                state["m"]=state["m"]*beta1+(1-beta1)*grad
                state["v"]=state["v"]*beta2+(1-beta2)*grad**2
                p.data -= lr_t*state["m"]/(torch.sqrt(state["v"])+eps) # Update weight tensor in-place.
                state["t"] = t  # Increment iteration number.
                
            
        return loss

def lr_cosine_schedule(it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int)-> float :
    
    if (it<warmup_iters):
        return it*max_learning_rate/warmup_iters
    
    elif(it<=cosine_cycle_iters):
        cos_number=math.cos(math.pi*(it-warmup_iters)/(cosine_cycle_iters-warmup_iters))

        return min_learning_rate+(max_learning_rate-min_learning_rate)*(1+cos_number)/2
    else:
        return min_learning_rate
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    
    eps=1e-6
    sum=0

    #Iterable可以是元组/list 和model.parameters(),如果是迭代器,那么遍历过一次就没了
    
    #创建列表容器,但是仍然共享一个数据,修改它的参数也会修改传入的参数
    parameters=list(parameters)

    for parameter in parameters:

        if parameter.grad == None:
            continue

        sum+=torch.sum(parameter.grad**2)

    p_l2_norm=torch.sqrt(sum).item()

    if(p_l2_norm>max_l2_norm):

        scale=max_l2_norm/(p_l2_norm+eps)

        for parameter in parameters:
            
            if parameter.grad is not None:
                parameter.grad*=scale

    return p_l2_norm

    
def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):

    #range不会采集到len(dataset)-context_length, 从这里开始的句子会恰好到末尾，所以它的target少一个，不能采

    #用range构建的数组太大了,这里直接用randint,给定上下界就可以采样
    start=np.random.randint(low=0,high=len(dataset)-context_length,size=batch_size)

    #利用广播避免循环

    start=start.reshape(-1,1)
    l=np.arange(context_length).reshape(1,-1)

    train_indices=start+l
    target_indices=start+l+1
    
    #存储的dataset是按uint16保存的,但是embedding要求要long作为索引
    train_dataset=torch.tensor(dataset[train_indices],dtype=torch.long)
    target_dataset=torch.tensor(dataset[target_indices],dtype=torch.long)

    return train_dataset.to(device),target_dataset.to(device)



def save_checkpoint(model : torch.nn.Module, optimizer : torch.optim.Optimizer, 
                    iteration :int, out :str | os.PathLike| typing.BinaryIO | typing.IO[bytes]):

    checkpoint_dict={"iteration":iteration,
                     "model":model.state_dict(),
                     "optimizer":optimizer.state_dict()}
    
    torch.save(checkpoint_dict,out)

def load_checkpoint(src : str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
                    model :torch.nn.Module, optimizer : torch.optim.Optimizer):
    
    obj=torch.load(src)

    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]


    
def validation(valid_data,model,eval_iters,batch_size,context_length,device):

    model.eval()

    total_loss = 0
    with torch.no_grad():
        for it in range(eval_iters):
            train_batch,target_batch =get_batch(valid_data,batch_size,context_length,device)
                
            predict_x_batch=model(train_batch)
            
            loss=cross_entropy(predict_x_batch.reshape(-1,predict_x_batch.shape[-1]),target_batch.flatten())
            total_loss+=loss.item()

    return total_loss/eval_iters
        


def train_script(train_path : str,valid_path : str, model :torch.nn.Module,
                 #记录输出的地址
                 train_log_file : str,val_log_file:str,best_model_path:str,save_model_path:str,
                 #记录频率
                 log_frequency:int ,eval_frequency:int,checkpoint_frequency:int,
                 #调整学习率的超参数
                 max_learning_rate: float,min_learning_rate: float,warmup_iters: int,cosine_cycle_iters: int,
                 #迭代次数
                 max_iterations:int ,eval_iterations:int,
                 #梯度裁剪的超参数
                 max_l2_norm : float,
                 #get batch的超参数
                 context_length :int,device:torch.device,batch_size=20,
                 #优化器的超参数
                 lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0,
                 ):
    
    #这时候文本已经转换成token ID
    train_data=np.memmap(
        filename=train_path,
        dtype="uint16",
        mode="r"
    )

    valid_data=np.memmap(
        filename=valid_path,
        dtype="uint16",
        mode="r"
    )

    
    # writer=SummaryWriter(summay_path)

    model.to(device=device)
    optimizer=AdamW(model.parameters(),lr,betas,eps,weight_decay)

    best_val_loss=float("inf")


    start_time=time.perf_counter()

    for iter in range(max_iterations):

        model.train()
        
        #设置学习率    
        cur_lr=lr_cosine_schedule(iter,max_learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters)
        #调整学习率
        for group in optimizer.param_groups :
            group["lr"]=cur_lr

        #清理旧梯度
        optimizer.zero_grad()

        #获取批量的训练数据和目标数据 shape:(batch_size,context_length)
        train_batch,target_batch =get_batch(train_data,batch_size,context_length,device)
        #输出是batch_size,context_length,vocab
        predict_x_batch=model(train_batch)
        #由于cross_entropy接收的是(batch_size,voacb_size 和 batch_size的输入,这里要变形)
        loss=cross_entropy(predict_x_batch.reshape(-1,predict_x_batch.shape[-1]),target_batch.flatten())
       

        #反向传播
        loss.backward()
        #进行梯度裁剪
        grad_after_norm=gradient_clipping(model.parameters(),max_l2_norm)
        #梯度更新
        optimizer.step()

        step=iter+1

        if step%log_frequency==0:
            if device.type=="cuda":
                torch.cuda.synchronize()
            cur_loss=loss.item()
            step_information={
                "step":step,
                "wall_clock":time.perf_counter()-start_time,
                "train_loss":cur_loss,
                "lr":cur_lr
            }

            wandb.log({
                "train/loss":cur_loss,
                "grad_after_norm":grad_after_norm,
                "cos_lr":cur_lr
            },step=step)
            # writer.add_scalar("Loss/train",cur_loss,step)
            # writer.add_scalar("grad_norm", grad_norm2, step)
            # writer.flush()
            with open(train_log_file,"a",encoding="utf-8") as f:
                json.dump(step_information,f)
                f.write("\n")
            
            print(cur_loss)
            

        if step%eval_frequency==0:
            val_loss=validation(valid_data,model,eval_iterations,batch_size,context_length,device)
            
            if device.type=="cuda":
                torch.cuda.synchronize()

            step_information={
                "step":step,
                "wall_clock":time.perf_counter()-start_time,
                "val_loss":val_loss,
            }
            wandb.log({
                "valid/loss":val_loss,
            },step=step)
            # writer.add_scalar("Loss/val",val_loss,step)
            # writer.flush()
            with open(val_log_file,"a",encoding="utf-8") as f:
                json.dump(step_information,f)
                f.write("\n")
            
            print(f"val loss :{val_loss}")

            

            if val_loss<best_val_loss:
                best_val_loss=val_loss

                save_pth=os.path.join(best_model_path,f"_{step}_{best_val_loss:.3f}.checkpoint")
                save_checkpoint(model,optimizer,step,save_pth)

        if step%checkpoint_frequency==0:
            save_pth=os.path.join(save_model_path,f"_{step}_{loss.item():.3f}.checkpoint")
            save_checkpoint(model,optimizer,step,save_pth)    

    
    # writer.close()

    

def sample_top_p(pred_p,top_p):

    #torch.sort会保留对应的索引,这里用降序
    pred_p_sort,indics=torch.sort(pred_p,descending=True)

    pred_sum = torch.cumsum(pred_p_sort, dim=0)

    #找那些在前top_p以内的pred
    target_pred = pred_sum < top_p

    #由于浮点数可能会使得最后一个的pred_sum<1，所以当top_p=1时，可能target_number=长度，导致越界
    if top_p<1:
        target_number=torch.sum(target_pred).item()
        target_pred[target_number]=True
        sample_token_p=pred_p_sort[target_pred]
    else:
        sample_token_p=pred_p_sort
    
    #获得那些在top_p以内的概率
    
    #重新归一化,此时不需要用softmax,直接除以综合
    sample_token_p=sample_token_p/torch.sum(sample_token_p)

    #返回sample_token_p内部的索引
    result_idx=torch.multinomial(sample_token_p,num_samples=1)

    return indics[result_idx]


def decoder(prompt :str ,maximum_token: int ,temperature: float ,top_p: float,model : torch.nn.Module, tokenizer,device="cpu"):

    generated_tokens=[]

    #应该是(context_length,) 要变成(1,context_length)
    prompt_tokens=torch.tensor(tokenizer.encode(prompt),device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():

        while len(generated_tokens)<maximum_token :
            
            if prompt_tokens.shape[1]>model.context_length:
                prompt_tokens=prompt_tokens[:,-model.context_length:]

            pred_logits=model(prompt_tokens)
            pred_p=softmax(pred_logits[0,-1,:],dim=-1,temperature=temperature)

            #这里返回的是tensor
            next_token_ID=sample_top_p(pred_p,top_p)

            generated_tokens.append(next_token_ID.item())
            prompt_tokens=torch.cat((prompt_tokens,next_token_ID.view(1,1)),dim=1)
            
            #如果是特殊字符就退出
            if tokenizer.decode([next_token_ID.item()]) == "<|endoftext|>":
                break

    return tokenizer.decode(generated_tokens)
        
            
