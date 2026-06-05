from cs336_basics.BPE import BPE_training
import pickle
input_path="D:/HuaweiMoveData/Users/huawei/Desktop/AI知识库/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
vocab_size=32000
special_tokens=["<|endoftext|>"]
vocab,merges=BPE_training(input_path,vocab_size,special_tokens)

with open("vocab_32000.pkl", "wb") as f:
    pickle.dump(vocab, f)

with open("merges_32000.pkl", "wb") as f:
    pickle.dump(merges, f)


