import torch.nn as nn
import torch
import einops
import math

from jaxtyping import Bool, Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
            Construct a lineartransformation module. 
            This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """

        super(Linear,self).__init__()
    
        std=math.sqrt(2/(in_features+out_features))
        weight=torch.empty(size=(out_features,in_features),dtype=dtype,device=device)
        #这里截断(-3std,3std)外的初始权重，重新采样
        self.weight=nn.Parameter(nn.init.trunc_normal_(weight,mean=0,std=std,a=-3*std,b=3*std))

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Apply the linear transformation to theinput.
        """
        #这里只堆最后一维度操作
        return einops.einsum(x,self.weight,"... d_in,d_out d_in -> ... d_out")
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
            Construct an embedding module.
            This function should accept the following parameters:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., 𝑑model
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super(Embedding,self).__init__()
        embedding_matrix=torch.empty(size=(num_embeddings,embedding_dim),device=device,dtype=dtype)
        self.embedding_matrix=nn.Parameter(nn.init.trunc_normal_(embedding_matrix,mean=0,std=1,a=-3,b=3))


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
            Lookup the embedding vectors for the given token IDs.
        """
        #直接取出每个token_ID对应的tensor
        return self.embedding_matrix[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
            Construct the RMSNorm module. This function should accept the following parameters:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """ 

        super(RMSNorm,self).__init__()
        self.d_model=d_model
        self.eps=eps

        self.gi=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Process an input tensor of shape (batch_size, sequence_length, d_model) 
            and return a tensor of the same shape.
        """
        #要计算平方
        x = x.to(torch.float32)
        mean_square=einops.reduce(x**2,"... d_model -> ...","sum")/self.d_model
        rms_x=torch.sqrt(mean_square + self.eps)

        return einops.einsum(x/einops.rearrange(rms_x,"...->... 1"),self.gi,"... d_model,d_model->... d_model")

def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:

    return in_features*torch.sigmoid(in_features)

class SwiGLU_FNN(nn.Module):
    def __init__(self, d_model: int, d_ff: int ,device=None, dtype=None):
        
        
        super(SwiGLU_FNN,self).__init__()
        self.w1=Linear(d_model,d_ff,device,dtype)
        self.w3=Linear(d_model,d_ff,device,dtype)

        self.w2=Linear(d_ff,d_model,device,dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        h=self.w3(x)

        x2=silu(self.w1(x))*h

        return self.w2(x2)
    
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
            Construct the RoPE module and create buffers if needed.
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be input
            device: torch.device | None = None Device to store the buffer on
        """
        super(RoPE,self).__init__()
        
        position=torch.arange(max_seq_len,device=device)
        dim_indices=torch.arange(0,d_k,2,device=device)
        theta_vector=pow(theta,-dim_indices/d_k)

        #不直接存R矩阵，因为太过于稀疏
        #max_seqlen,d_k/2
        angles=torch.outer(position,theta_vector)

        #属于这个模型,但不是训练的参数
        self.register_buffer("cos_angles",torch.cos(angles),persistent=False)
        self.register_buffer("sin_angles",torch.sin(angles),persistent=False)


        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
            Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
            that you should tolerate 𝑥 with an arbitrary number of batch dimensions. You should assume
            that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
            𝑥 along the sequence dimension.
        """
        
        cos_need=self.cos_angles[token_positions]#(...，seq_len,d_k/2)
        sin_need=self.sin_angles[token_positions]

        x_even=x[...,0::2]
        x_odd=x[...,1::2]

        rotate_x_even=cos_need*x_even-sin_need*x_odd
        rotate_x_odd=sin_need*x_even+cos_need*x_odd

        result=torch.empty_like(x)
        result[...,0::2]=rotate_x_even
        result[...,1::2]=rotate_x_odd

        return result




def softmax(x: torch.Tensor,dim: int ,temperature =1) -> torch.Tensor:
    #会返回value,indices 

    x=x/temperature

    x_max=torch.max(x,dim=dim,keepdim=True).values
  
    x=x-x_max
    exp_x=torch.exp(x)

    return exp_x/torch.sum(exp_x,dim=dim,keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None):
    
    d_k=Q.shape[-1]

    Attention_score=einops.einsum(Q,K,"... queries d_k,... keys d_k->... queries keys")/math.sqrt(d_k)

    if mask is not None:
        #如果不指定类型,mask_score会变成Bool
        mask_score=torch.zeros_like(mask,dtype=torch.float32)
        mask_score[~mask]=float("-inf")
        Attention_score+=mask_score

    return einops.einsum(softmax(Attention_score,dim=-1),V,"... queries keys, ... keys d_v -> ... queries d_v")



class Multi_head_Attention(nn.Module):
    def __init__(self,d_model,num_heads):
        """
            设置一个投影矩阵,大小为d_model,num_heads*d_v,这里d_v=d_model/num_heads
        """
        super(Multi_head_Attention,self).__init__()
        std=math.sqrt(1/d_model)


        self.weight=nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model,d_model),mean=0,std=std,a=-3*std,b=3*std))
        self.num_heads=num_heads

        self.q_weight=nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model,d_model),mean=0,std=std,a=-3*std,b=3*std))
        self.k_weight=nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model,d_model),mean=0,std=std,a=-3*std,b=3*std))
        self.v_weight=nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model,d_model),mean=0,std=std,a=-3*std,b=3*std))

    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"])-> Float[Tensor, " ... sequence_length d_model"]:
        

        Q=einops.einsum(in_features,self.q_weight,"... sequence_length d_model,h_d_q d_model ->... sequence_length h_d_q")
        K=einops.einsum(in_features,self.k_weight,"... sequence_length d_model,h_d_k d_model ->... sequence_length h_d_k")
        V=einops.einsum(in_features,self.v_weight,"... sequence_length d_model,h_d_v d_model ->... sequence_length h_d_v") 


        n_seq=in_features.shape[-2]

        mask=torch.tril(torch.ones((n_seq,n_seq),dtype=torch.bool,device=in_features.device))

        Multi_Q=einops.rearrange(Q," ... sequence_length (num_heads d_q) -> ... num_heads sequence_length d_q",num_heads=self.num_heads)
        Multi_K=einops.rearrange(K," ... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k",num_heads=self.num_heads)
        Multi_V=einops.rearrange(V," ... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v",num_heads=self.num_heads)

        result=scaled_dot_product_attention(Multi_Q,Multi_K,Multi_V,mask) #
        concat_result=einops.rearrange(result,"... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)")

        return einops.einsum(self.weight,concat_result,"d_model num_heads_d_v, ... num_heads_d_v -> ... d_model")
    
class Multi_head_Attention_with_RoPE(nn.Module):
    def __init__(self,d_model,num_heads,theta,max_seq_len):
        """
            初始化权重的时候要注意,不能初始化为zeros,不然会没法更新,采用随机初始化的方法这里
        """
        super(Multi_head_Attention_with_RoPE,self).__init__()

        std=math.sqrt(1/d_model)

        assert d_model%num_heads ==0 , "d_model 必须是num_heads的整数倍"

        self.weight=nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model,d_model),mean=0,std=std,a=-3*std,b=3*std))
        self.num_heads=num_heads
        self.rope=RoPE(theta,d_model//num_heads,max_seq_len)

        self.q_weight=nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model,d_model),mean=0,std=std,a=-3*std,b=3*std))
        self.k_weight=nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model,d_model),mean=0,std=std,a=-3*std,b=3*std))
        self.v_weight=nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(d_model,d_model),mean=0,std=std,a=-3*std,b=3*std))

      

    
    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"])-> Float[Tensor, " ... sequence_length d_model"]:
        """
            
        """
        Q=einops.einsum(in_features,self.q_weight,"... sequence_length d_model,h_d_q d_model ->... sequence_length h_d_q")
        K=einops.einsum(in_features,self.k_weight,"... sequence_length d_model,h_d_k d_model ->... sequence_length h_d_k")
        V=einops.einsum(in_features,self.v_weight,"... sequence_length d_model,h_d_v d_model ->... sequence_length h_d_v") 


        n_seq=in_features.shape[-2]

        #创建的临时张量要保证在同一个设备上
        token_position=torch.arange(0,n_seq,1,device=in_features.device)
        

        mask=torch.tril(torch.ones((n_seq,n_seq),dtype=torch.bool,device=in_features.device))

        Multi_Q=einops.rearrange(Q," ... sequence_length (num_heads d_q) -> ... num_heads sequence_length d_q",num_heads=self.num_heads)
        Multi_K=einops.rearrange(K," ... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k",num_heads=self.num_heads)
        Multi_V=einops.rearrange(V," ... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v",num_heads=self.num_heads)
        
        #别用for循环，效率低
        # head_result=[]
        # for i  in range(self.num_heads):
        #     R_q=self.rope(Multi_Q[...,i,:,:],token_position)
        #     R_k=self.rope(Multi_K[...,i,:,:],token_position)
        #     head_result.append(scaled_dot_product_attention(R_q,R_k,Multi_V[...,i,:,:],mask))

        R_q=self.rope(Multi_Q,token_position)
        R_k=self.rope(Multi_K,token_position)

        result=scaled_dot_product_attention(R_q,R_k,Multi_V,mask) #
        concat_result=einops.rearrange(result,"... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)")

        return einops.einsum(self.weight,concat_result,"d_model num_heads_d_v, ... seq_l num_heads_d_v -> ... seq_l d_model")
    
class Transformer_block(nn.Module):
    def __init__(self, d_model: int,num_heads: int,d_ff: int,theta: float,max_seq_len: int):
        super(Transformer_block,self).__init__()

        self.rmsnorm1=RMSNorm(d_model)
        self.MHA=Multi_head_Attention_with_RoPE(d_model,num_heads,theta,max_seq_len)


        self.rmsnorm2=RMSNorm(d_model)
        self.ffn=SwiGLU_FNN(d_model,d_ff)

    def forward(self,
                in_features: Float[Tensor, " batch sequence_length d_model"]
                )-> Float[Tensor, " batch sequence_length d_model"]:
        

        # x1=in_features+self.MHA(self.rmsnorm1(in_features))

        # x2=x1+self.ffn(self.rmsnorm2(x1))
        #对比postnorm和prenorm
        x1=self.rmsnorm1(in_features+self.MHA(in_features))

        x2=self.rmsnorm2(x1+self.ffn(x1))
        return x2
    

class Transformer_lm(nn.Module):
    def __init__(self, vocab_size: int,context_length: int,num_layers: int,d_model: int,
                 num_heads: int,d_ff: int,rope_theta: float,seed = 2026):

        super(Transformer_lm,self).__init__()
        self.embedding=Embedding(num_embeddings=vocab_size,embedding_dim=d_model)

        self.context_length=context_length

        self.layers=nn.ModuleList()

        for num in range(num_layers):
            self.layers.append(Transformer_block(d_model,num_heads,d_ff,theta=rope_theta,max_seq_len=context_length))

        self.rmsnorm=RMSNorm(d_model)
        self.linear_final=Linear(in_features=d_model,out_features=vocab_size)
        

    def forward(self,in_indices: Int[Tensor, " batch_size sequence_length"])-> Float[Tensor, " batch_size sequence_length vocab_size"]:

        x=self.embedding(in_indices)

        for layer in self.layers:
            x=layer(x)

        x_norm=self.rmsnorm(x)
        logits= self.linear_final(x_norm)
        
        return logits
