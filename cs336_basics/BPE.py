import regex,re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter,defaultdict
import pickle
from multiprocessing import Pool


"""
    预编译,这样就不用每次count_text的时候都编译一遍,count_text会被调用非常多次毕竟
    PAT是gpt的正则化方法
"""

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PATTERN=regex.compile(PAT)

def count_text(text:str,all_counter):
    """
    输入参数:
        text:传入的内容
        all_counter:统计全局的字典,直接更新,而不是再生成一个
    输出参数:
        word_count:返回一个计数字典
    -----
    这里直接传入all_counter对象更新,这样就不用创建很多小的Counter
    """
    all_counter.update(match.group() for match in PATTERN.finditer(text))


def worker(input_path,start,end,special_tokens_escape):
    """
        每个进程的worker部分
        -----
        如果是window的训练集,需要先对chunk进行替换
        因为Unix中的换行在window中是\r\n,它会被优先配对,抢占其他pair的机会
        而测试是按照Unix的换行,这样测试就过不了
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        
        chunk=chunk.replace("\r\n","\n")
        chunk=chunk.replace("\r","\n")
        chunk_list=re.split("|".join(special_tokens_escape),chunk)

        chunk_counter=Counter()
        for chunk_text in chunk_list:
            count_text(chunk_text,chunk_counter)

    return chunk_counter

def parallel_pre_tokenization(input_path:str,special_tokens_escape):
    """ 
        并行化处理pre-tokenization,因为通过cProfile查看这里是大头(在vocab_size=2000时)!
        -----
        只给子进程传入文件路径和要读的位置,而不是直接传入文件

    """

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        tasks=[]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((input_path,start,end,special_tokens_escape))
        
        all_counter=Counter()
        with Pool(processes=num_processes) as pool:
            result=pool.starmap(worker,tasks)
            for chunk_counter in result:
                all_counter.update(chunk_counter)
            

    return all_counter

def pre_tokenization(input_path:str,special_tokens_escape):
    """
        Pre-tokenization部分.
        对单词进行去重后,转为byte级别的元组,{(1,342,13..):12,...}统计每个word的出现次数
    """
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        all_counter=Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
       
            chunk=chunk.replace("\r\n","\n")
            chunk=chunk.replace("\r","\n")
            chunk_list=re.split("|".join(special_tokens_escape),chunk)

            for chunk_text in chunk_list:
                count_text(chunk_text,all_counter)
        
    return all_counter

def merge(pair_to_word_idx,byte_counter_list,pair_count,pair,new_idx,
          count_to_pairs,max_count):
    """
    传入参数:
        pair_to_word_idx:pair对应相关的idx,idx对应byte_counter_list的索引
        byte_counter_list:列表,存储着(tuple_word,count)元组
        pair_count: pair对应的次数,在merge时需要更新
        pair:找到的频率最高的pair
        idx:pair对应的符号,
        count_to_pairs:字典,key是次数,对应出现那么多次的pair
        max_count:当前出现最多的pair
    -----
    pair_to_word_idx是用来维护:pair出现在哪些word中,这样当pair被修改时,只有它们会受影响
    
    当找到这些受影响word的idx后,通过byte_counter_list找到 对应的具体已经变成元组的word,和它们的出现次数count
    eg:
        (a,b,c,d): 3

    然后依次遍历(a,b) (b,c) (c,d) ,完成:
        1.减去旧的pair_count计数,比如原来(a,b): 5, 现在(a,b):2
        2.更改count_to_pairs: 先从原来的5的计数中删去(a,b) 然后在新的计数2中加上(a,b)
        3.从pair_to_word_idx中先斩断它们和当前word的联系

    然后合并:
       (a,x,d)

    再依次遍历,完成:

        1.如果当前pair存在, eg: (a,x):10, 则删去它的目前count10中与(a,x)的联系
        2.更新pair_count的值,(a,x)=原来count+3
        3.用更新后的count添加与(a,x)的联系
        4.从pair_to_word_idx中添加pair与当前word的联系
    
    """
    
    byte_idx_list=pair_to_word_idx[pair].copy()
    
    for idx in byte_idx_list:
        tuple_word,count=byte_counter_list[idx]
        #先减去旧的pair
        for old_pair in zip(tuple_word,tuple_word[1:]):
            #减去旧的计数
            old_count=pair_count[old_pair]
            count_to_pairs[old_count].remove(old_pair)

            pair_count[old_pair]-=count

            if pair_count[old_pair]>0:
                count_to_pairs[pair_count[old_pair]].add(old_pair)
            else:
                pair_count.pop(old_pair)

            #减去旧的联系，pair_to_word_idx里存放的是集合
            pair_to_word_idx[old_pair].discard(idx)

        #merge 新的pair
        tmp=[]
        i=0
        while i < len(tuple_word):
            if (i<len(tuple_word)-1) and tuple_word[i]==pair[0] and tuple_word[i+1]==pair[1]:
                tmp.append(new_idx)
                i+=2
                continue
            else:
                tmp.append(tuple_word[i])
            i+=1
        new_tuple_word=tuple(tmp)
        byte_counter_list[idx]=(new_tuple_word,count)
        
        

        #加上新的pair
        for new_pair in zip(new_tuple_word,new_tuple_word[1:]):
            if pair_count.get(new_pair):
                count_to_pairs[pair_count[new_pair]].remove(new_pair)

            pair_count[new_pair]=pair_count.get(new_pair,0)+count

            new_count=pair_count[new_pair]
            if(new_count>max_count):
                max_count=new_count

            count_to_pairs[new_count].add(new_pair)

            pair_to_word_idx[new_pair].add(idx)




   
        
    return pair_to_word_idx,byte_counter_list,pair_count,max_count


def BPE_training(input_path:str,vocab_size:int,special_tokens:list[str]):
    """
    输入参数:
        input_path:BPE训练的text输入位置
        vocab_size:最大词表数量,包括初始字节词汇,合并后的字节词汇,和特殊词汇
        special_tokens:字节词汇不能越过这些special_tokens合并,它们也不会被计入合并统计
    输出参数:
        ## Usage
with open(..., "rb") as f:
    vocab: dict[int, bytes] 将字节映射回字节
    merges: list[tuple[bytes, bytes]] 返回合并的规则,需要按照创建的顺序排列
    """

    merges=[]
    vocab={}

    """
        初始化vocab,添加0-255字符和special_tokens
    """
    for i in range(256):
        vocab[i]=bytes([i])

    for special_token in special_tokens:
        vocab[len(vocab)]=special_token.encode("utf-8")

    
    
    #这里需要先进行安全转义，因为special tokens 里面含有 | 等字符，不利于正则化
    special_tokens_escape=[re.escape(special_token) for special_token in special_tokens]

    
    all_counter=parallel_pre_tokenization(input_path,special_tokens_escape)
    # all_counter=pre_tokenization(input_path,special_tokens_escape)

    """
        byte_counter: 字典
        {
            (1,234,56,...):出现频率,
            ...
        }
        每次找到pair后,更新它的key值
    """
    byte_counter={}
    for word,count in all_counter.items():
        tuple_word=tuple(b for b in word.encode("utf-8"))
        byte_counter[tuple_word]=count

    """
        将byte_counter转化成列表,这样就可以把每一个pair与一个索引相互对应
    """
    byte_counter_list=list(byte_counter.items())

    
    """
        pair_to_word_idx存储pair对应的word下标,下标通过查找byte_counter_list查看对应的tuple
        用defaultdict默认类型
    """
    pair_to_word_idx=defaultdict(set)

    """
        pair_count用来统计每一对pair的出现次数
    """
    pair_count={}
    for i in range(len(byte_counter_list)):

        tuple_byte=byte_counter_list[i][0]
        count=byte_counter_list[i][1]

        for pair in zip(tuple_byte,tuple_byte[1:]):
            pair_count[pair]=pair_count.get(pair,0)+count
            #存储每一个pair对应多少word
            pair_to_word_idx[pair].add(i)

    #用count_to_pairs统计每个count到底有哪些pair
    count_to_pairs=defaultdict(set)

    max_count=0
    for pair,count in pair_count.items():
        count_to_pairs[count].add(pair)
        max_count=max(max_count,count)

    

    while len(vocab)<vocab_size:


        while max_count > 0 and not count_to_pairs[max_count]:
            max_count -= 1

        if max_count==0:
            break
        # 依次比较出现次数，第一个字母的字典序，第二个字母的字典序，注意这里的pair[0] [1]都是idx,要转换成byte比较
        most_common_pairs=count_to_pairs[max_count]

        most_common_pair=max(most_common_pairs,
                             key=lambda pair:(vocab[pair[0]],vocab[pair[1]])
                        )

        #更新merges
        merges.append((vocab[most_common_pair[0]],vocab[most_common_pair[1]]))
        #更新vocab
        idx=len(vocab)
        new_byte=(vocab[most_common_pair[0]]+vocab[most_common_pair[1]])
        vocab[idx]=(new_byte)
        """
            合并pair。
            1旧的pair在pair_to_word_idx中需要删去它对应的tuple_byte,新组合的pair需要被添加
            2.pair_count的计数方式需要改动
            3.byte_counter_list需要改动有pair出现的idx,更新对应的key
        """
       
        pair_to_word_idx,byte_counter_list,pair_count,max_count=merge(pair_to_word_idx,byte_counter_list,pair_count,most_common_pair,idx
                                                            ,count_to_pairs,max_count)
        
        

    return vocab,merges

def main():
    input_path="D:/HuaweiMoveData/Users/huawei/Desktop/AI知识库/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size=10000
    special_tokens=["<|endoftext|>"]
    vocab,merges=BPE_training(input_path,vocab_size,special_tokens)

    with open("vocab_tiny.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open("merges_tiny.pkl", "wb") as f:
        pickle.dump(merges, f)

if __name__ == '__main__':
    main()