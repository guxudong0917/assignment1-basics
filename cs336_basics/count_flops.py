def count(layer,n_seq,d_model,vocab_size=50527):

    #由于mha部分有n_seq二次项，所以增加上下文，mha比例提升
    transformer_block_mha=8*n_seq*pow(d_model,2)+4*pow(n_seq,2)*d_model
    #如果模型变大,d_model部分会使得ffn比例提升
    transformer_block_ffn=16*n_seq*pow(d_model,2)

    mha=layer*transformer_block_mha
    ffn=layer*transformer_block_ffn

    linear=2*n_seq*d_model*vocab_size

    sum=mha+ffn+linear
    print(layer)
    print(f"mha 占比:{mha/sum}")
    print(f"ffn 占比:{ffn/sum}")

    print(f"FLOPS: {sum}")
    

# count(12,1024,768)
# count(24,1024,1024)
# count(36,1024,1280)

# count(48,16384,1600)

from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
                return loss
    
# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1000)
# for t in range(10):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.
