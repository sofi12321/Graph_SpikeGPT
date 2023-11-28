import numpy as np
import os, sys, types
import torch
from torch import nn


def prepare_env():
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    except:
        pass
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    args = types.SimpleNamespace()

    args.RUN_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 'cuda' // 'cpu' (already fast)
    args.FLOAT_MODE = "fp32"  # fp16 (good for GPU, does not work for CPU) // fp32 (good for CPU) // bf16 (less accurate, but works for CPU)
    os.environ[
        "RWKV_JIT_ON"] = '1'  # '1' or '0'. very useful for GPU/CPU fp32, but might be harmful for GPU fp16. please benchmark !!!
    vocab_size = 50277

    # MODEL_NAME = model_path + 'SpikeGPT-216M'
    n_layer = 18
    n_embd = 768
    ctx_len = 1024

    args.MODEL_NAME = 'SpikeGPT-216M'

    # args.MODEL_NAME = MODEL_NAME
    args.n_layer = n_layer
    args.n_embd = n_embd
    args.ctx_len = ctx_len
    args.vocab_size = vocab_size
    args.head_qk = 0
    args.pre_ffn = 0
    args.grad_cp = 0
    args.my_pos_emb = 0
    os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE
    return args


def load_embedding_weights(model_path, args):
    # Return nn.Embedding with weights from SpikeGPT embedding
    MODEL_NAME = model_path + 'SpikeGPT-216M'
    args.MODEL_NAME = MODEL_NAME

    # Load pretrained state
    model = RWKV_RNN(args)

    # Get embedding layer from the model
    emb = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.n_embd)
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
    # print("Number of tokens:", len(ctx))

    return emb(torch.tensor(ctx))


def tokenize(text, tokenizer):
    if type(text) == str:
        text = text.lower()
    if tokenizer.charMode:
        context = tokenizer.refine_context(text)
        ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
    else:
        ctx = tokenizer.tokenizer.encode(text)
    return ctx


if __name__ == "__main__":
    # Clone SpikeGPT to the src.third_party folder
    args = prepare_env()
    from scr.third_party.SpikeGPT.src.utils import TOKENIZER
    from scr.third_party.SpikeGPT.src.model_run import RWKV_RNN

    emb = load_embedding_weights("/content/SpikeGPT-OpenWebText-216M/", args)
    tokenizer = load_tokenizer()
