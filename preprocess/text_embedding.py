% !git clone https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M
% !git clone https://github.com/ridgerchu/SpikeGPT.git
% !pip install torch matplotlib numpy tqdm torchvision scipy ninja accelerate transformers
% %cd SpikeGPT

import numpy as np
import math, os, sys, types, time, gc
import torch
from src.utils import TOKENIZER
from src.model_run import RWKV_RNN
import matplotlib.ticker as ticker
from torch import nn

def load_embedding_weights(model_path):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    except:
        pass
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    args = types.SimpleNamespace()

    args.RUN_DEVICE = "cpu" # 'cuda' // 'cpu' (already fast)
    args.FLOAT_MODE = "fp32" # fp16 (good for GPU, does not work for CPU) // fp32 (good for CPU) // bf16 (less accurate, but works for CPU)
    os.environ["RWKV_JIT_ON"] = '1' # '1' or '0'. very useful for GPU/CPU fp32, but might be harmful for GPU fp16. please benchmark !!!    
    vocab_size = 50277

    MODEL_NAME = model_path + 'SpikeGPT-216M'
    n_layer = 18
    n_embd = 768
    ctx_len = 1024

    args.MODEL_NAME = MODEL_NAME
    args.n_layer = n_layer
    args.n_embd = n_embd
    args.ctx_len = ctx_len
    args.vocab_size = vocab_size
    args.head_qk = 0
    args.pre_ffn = 0
    args.grad_cp = 0
    args.my_pos_emb = 0
    os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE

    # Load pretrained state
    model = RWKV_RNN(args)

    # Get embedding layer from the model
    emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
    # emb.weight.data = torch.rand(num_node,embedding_dim)
    # print(emb.weight.data)
    emb.weight.data = model.w.emb.weight.data

    return emb

def load_tokenizer():
    TOKEN_MODE = "pile"
    WORD_NAME = [
        "20B_tokenizer.json",
        "20B_tokenizer.json",
    ]  # [vocab, vocab] for Pile model
    UNKNOWN_CHAR = None

    tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
    if TOKEN_MODE == "pile":
        assert tokenizer.tokenizer.decode([187]) == '\n'
    return tokenizer


def transform_text_pretrained_embedding(text, emb, tokenizer):
    if tokenizer.charMode:
        context = tokenizer.refine_context(text)
        ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in text]
    else:
        ctx = tokenizer.tokenizer.encode(text)
    print("Number of tokens:", len(ctx))

    return emb(torch.tensor(ctx))

emb = load_embedding_weights("/content/SpikeGPT-OpenWebText-216M/")
tokenizer = load_tokenizer()
text = "In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet."
transform_text_pretrained_embedding(text, emb, tokenizer)
