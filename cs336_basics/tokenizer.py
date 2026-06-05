
import pickle,regex,re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PATTERN=regex.compile(PAT)

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        """
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        -----
            1.合并时将byte合并为byte,而encode时需要输出id,所以这里要反向一个vocab,存储bytes->int

            2.合并时需要记住每个pair的顺序,可以用字典存储,然后找到最靠前的一对
        """
        self.vocab=vocab
        #反向存储
        self.cache={}
        self.max_cache_size=1000000


        self.bytes_2id_vocab={bytes_:id for id,bytes_ in vocab.items()}
        #存储每个合并pair的优先度
        self.merge_rank={merge_pair:i  for i,merge_pair in enumerate(merges)}
        if special_tokens:
            self.special_tokens=sorted(special_tokens,
                                   key= lambda x:len(x),reverse=True)

            for special_token in self.special_tokens:
                byte_special_token=special_token.encode("utf-8")
                if self.bytes_2id_vocab.get(byte_special_token)==None:
                    self.vocab[len(self.bytes_2id_vocab)]=byte_special_token
                    self.bytes_2id_vocab[byte_special_token]=len(self.bytes_2id_vocab)
        else:
            self.special_tokens=special_tokens

        

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None): 
        """
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        with open(vocab_filepath,"rb") as f:
            vocab=pickle.load(f)

        with open(merges_filepath,"rb") as f:
            merges=pickle.load(f)

        return cls(vocab,merges,special_tokens)

    def merge(self,pretoken):
        
        if self.cache.get(pretoken) is not None:
            return self.cache[pretoken]


        tuple_pretoken=tuple(bytes([b]) for b in pretoken.encode("utf-8"))

        while (True):
            best_pair_rank=len(self.merge_rank)
            best_pair=None
            for pair in zip(tuple_pretoken,tuple_pretoken[1:]):
                if self.merge_rank.get(pair)!=None and best_pair_rank>self.merge_rank[pair]:
                    
                    best_pair_rank=self.merge_rank[pair]
                    best_pair=pair

            if best_pair==None:
                break
            
            new_tuple_pretoken=[]
            i=0
            while i<len(tuple_pretoken):
                if(i<len(tuple_pretoken)-1) and tuple_pretoken[i]==best_pair[0] and tuple_pretoken[i+1]==best_pair[1]:
                    new_tuple_pretoken.append(best_pair[0]+best_pair[1])
                    i+=2
                else:
                    new_tuple_pretoken.append(tuple_pretoken[i])
                    i+=1
            tuple_pretoken=tuple(new_tuple_pretoken)

        if len(self.cache) < self.max_cache_size:
            self.cache[pretoken] = tuple(self.bytes_2id_vocab[b] for b in tuple_pretoken)
            return self.cache[pretoken]

        return tuple(self.bytes_2id_vocab[b] for b in tuple_pretoken)
     
    def encode(self, text: str) -> list[int]:
        """
            Encode an input text into a sequence of token IDs.
        """
        token_IDs=[]
        # 现在就是(special_token) , 这样split special_token也会被放进去

        if self.special_tokens !=None:
            special_tokens_escape=[re.escape(special_token) for special_token in self.special_tokens]
            pattern="("+"|".join(special_tokens_escape)+")"
            split_texts=re.split(pattern,text)

            for split_text in split_texts:
                #如果是特殊字符,直接处理
                if split_text in self.special_tokens:
                    token_IDs.append(self.bytes_2id_vocab[split_text.encode("utf-8")])
                else:
                #如果不是,则逐个处理每个词
                    for match in PATTERN.finditer(split_text):
                        pretoken=match.group()
                        #pretoken.encode("utf-8") 得到的是b"speical" 遍历得到的是int数字,要再转成byte
                       
                        merge_token=self.merge(pretoken)
                        token_IDs.extend(merge_token)

        else:
            for match in PATTERN.finditer(text):
                        pretoken=match.group()
                      
                        merge_token=self.merge(pretoken)
                        token_IDs.extend(merge_token)

        return token_IDs
                
                
         
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory.

        这里不显示传入字符,而是传入一个字符串迭代器,比如 with open() as f: f就是迭代器 

        返回迭代器,用yield  类似return, 但是分批返回

        这里要求一个ID一个ID返回
        """

        for chunk in iterable:
            token_IDs=self.encode(chunk)
            for token_ID in token_IDs:
                yield token_ID

    def decode(self, ids: list[int]) -> str:
        """
            Decode a sequence of token IDs into text.
            To test your Tokenizer against our provided tests, you will first need to implement the test
            adapter at [adapters.get_tokenizer] . Then, run uv run pytest tests/test_tokenizer.py. Your
            implementation should be able to pass all tests.
        """

        result_bytes=b""
        for id in ids:
            result_bytes+=self.vocab[id]
        
        return result_bytes.decode("utf-8",errors="replace")

def main():
    merges_path="merges_tiny.pkl"
    vocab_path="vocab_tiny.pkl"
    tokenizer=Tokenizer.from_files(merges_filepath=merges_path,vocab_filepath=vocab_path,special_tokens=["<|endoftext|>"])
    input_path="D:/HuaweiMoveData/Users/huawei/Desktop/AI知识库/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"

    all_ids=[]
    #读取bytes解码成str
    with open(input_path,"r",encoding="utf-8") as f:
         for id in tokenizer.encode_iterable(f):
            all_ids.append(id)

    with open("tiny_story_ids.pkl", "wb") as f:
        pickle.dump(all_ids, f)


if __name__=="__main__":
    main()