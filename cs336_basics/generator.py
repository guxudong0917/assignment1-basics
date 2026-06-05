from cs336_basics.transformer_train import decoder,cross_entropy
from cs336_basics.Transformer import Transformer_lm
from cs336_basics.tokenizer import Tokenizer
import torch
import numpy as np

model=Transformer_lm(vocab_size=32000,context_length=256,num_layers=4,d_model=512,num_heads=16,d_ff=1344,rope_theta=10000)


device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

checkpoint=torch.load("./result/batch64_owt_full_max_lr1e-3/checkpoint/best_model/_19500_4.070.checkpoint",
                      map_location=device)
model.load_state_dict(checkpoint["model"])

model.to(device)

vocab_path="./tokenizer_data/vocab_owt.pkl"
merges_path="./tokenizer_data/merges_owt.pkl"
special_tokens=["<|endoftext|>"]

tokenizer=Tokenizer.from_files(vocab_filepath=vocab_path,merges_filepath=merges_path,special_tokens=special_tokens)


str="Once upon a time there was a little boy named Ben. Ben loved to explore the world around him.\
He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben\
 was walking through the store when he came across a very special vase. When Ben saw it he was\
 amazed! He said, "
str1="hi, today is a good day"

str3 = """Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan
Judge.
Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, “no corner of American
culture is more precisely counted, more passionately quantified, than performances of baseball
players.” With a few clicks here and there, you can find out that Noah Syndergaard’s fastball
revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the
game’s highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that
seem ripped from a video game or science fiction novel. The rising ocean of data has empowered
an increasingly important actor in baseball’s culture: the analytical hobbyist.
That empowerment comes with added scrutiny – on the measurements, but also on the people
and publications behind them. With Baseball Prospectus, Pavlidis knew all about the backlash
that accompanies quantitative imperfection. He also knew the site’s catching metrics needed to be
reworked, and that it would take a learned mind – someone who could tackle complex statistical
modeling problems – to complete the job.
“He freaks us out.” Harry Pavlidis
Pavlidis had a hunch that Judge “got it” based on the latter’s writing and their interaction at a
site-sponsored ballpark event."""
# prompt_ids = tokenizer.encode(str)
# print(prompt_ids)

prompt = "Machine learning is a field of computer science that focuses on building systems that can learn from data. In recent years, deep learning has become especially popular because"

prompt = "The most surprising thing about the project was not that it failed, but that everyone involved seemed to learn something useful from it. At first,"
result=decoder(prompt=prompt,maximum_token=256,temperature=0.9,top_p=0.95,model=model,tokenizer=tokenizer,device=device)

print(f"prompt:{prompt}")

print(f"generator:{result}")

