from cs336_basics.transformer_train import decoder,cross_entropy
from cs336_basics.Transformer import Transformer_lm
import torch
import numpy as np

model=Transformer_lm(vocab_size=10000,context_length=256,num_layers=4,d_model=512,num_heads=16,d_ff=1344,rope_theta=10000)

checkpoint=torch.load("/root/autodl-tmp/cs336_basics/result/batch64_tiny_max_lr2e-3/checkpoint/best_model/_20000_1.360.checkpoint")
model.load_state_dict(checkpoint["model"])

valid_data=np.memmap(
    filename="/root/autodl-tmp/data/TinyStoriesV2-GPT4-valid.bin",
    dtype="uint16",
    mode="r"
)

device=torch.device("cuda")
model.to(device)

def validation_full(valid_data, model, batch_size, context_length, device):
    model.eval()
    losses = []

    # 每个样本需要 context_length 个输入 + 1 个 target
    num_chunks = (len(valid_data) - 1) // context_length

    with torch.no_grad():
        for start in range(0, num_chunks, batch_size):
            actual_bs = min(batch_size, num_chunks - start)

            xs = []
            ys = []

            for b in range(actual_bs):
                idx = (start + b) * context_length
                x = torch.tensor(
                    valid_data[idx : idx + context_length],
                    dtype=torch.long
                )
                y = torch.tensor(
                    valid_data[idx + 1 : idx + context_length + 1],
                    dtype=torch.long
                )
                xs.append(x)
                ys.append(y)

            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)

            logits = model(x)
            loss = cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1)
            )

            # 按实际 token 数加权
            losses.append((loss.item(), actual_bs * context_length))

    total_loss = sum(l * n for l, n in losses)
    total_tokens = sum(n for _, n in losses)

    return total_loss / total_tokens

loss=validation_full(valid_data,model,batch_size=32,context_length=256,device=torch.device("cuda"))
print(loss)