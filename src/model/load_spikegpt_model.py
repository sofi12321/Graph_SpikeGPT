import gc
import torch
import time
import numpy as np

# Clone SpikeGPT to the src.third_party folder before

def get_spike_gpt_model(args):
    # Load model from third party
    from scr.third_party.SpikeGPT.src.model_run import RWKV_RNN

    model = RWKV_RNN(args)

    print(f'\nOptimizing speed...')

    gc.collect()
    torch.cuda.empty_cache()
    return model


def run_spike_gpt(model, context, ctx, args, tokenizer, num_trials=2, length_per_trial = 100):
    # Code is based on SpikeGPT run.py, but changed for this project
    # To be able to use with the created model
    # ctx = sequence of input tokens
    # Context - real sentence
    ctx_len = args.ctx_len

    TOKEN_MODE = "pile"

    src_len = len(ctx)
    src_ctx = ctx.copy()

    TEMPERATURE = 1.5
    top_p = 0.7
    top_p_newline = 0.9  # only used in TOKEN_MODE = char

    DEBUG_DEBUG = False  # True False --> show softmax output

    print("\nYour prompt has " + str(src_len) + " tokens.")

    time_slot = {}
    time_ref = time.time_ns()

    def record_time(name):
        if name not in time_slot:
            time_slot[name] = 1e20
        tt = (time.time_ns() - time_ref) / 1e9
        if tt < time_slot[name]:
            time_slot[name] = tt

    init_state = None
    init_out = None
    state = None
    mem1 = None
    mem2 = None
    out = None

    for TRIAL in range(1 if DEBUG_DEBUG else num_trials):
        print(("-" * 50) + '\n' + context)

        time_ref = time.time_ns()
        ctx = src_ctx.copy()

        if TRIAL == 0:
            for i in range(src_len):
                x = ctx[: i + 1]
                if i == src_len - 1:
                    init_out, init_state, mem1, mem2 = model.forward(x, init_state, mem1, mem2)
                else:
                    init_state, mem1, mem2 = model.forward(x, init_state, mem1, mem2, preprocess_only=True)
            gc.collect()
            torch.cuda.empty_cache()

        record_time('preprocess')
        out_last = src_len
        for i in range(src_len, src_len + (1 if DEBUG_DEBUG else length_per_trial)):
            x = ctx[: i + 1]
            x = x[-ctx_len:]

            if i == src_len:
                out = init_out.clone()
                state = init_state.clone()
            else:
                out, state, mem1, mem2 = model.forward(x, state, mem1, mem2)
            if DEBUG_DEBUG:
                print("model", np.array(x), "==>", np.array(out), np.max(out.cpu().numpy()), np.min(out.cpu().numpy()))
            if TOKEN_MODE == "pile":
                out[0] = -999999999  # disable <|endoftext|>

            ttt = tokenizer.sample_logits(
                out,
                x,
                ctx_len,
                temperature=TEMPERATURE,
                top_p_usual=top_p,
                top_p_newline=top_p_newline,
            )
            ttt = int(ttt)
            ctx += [ttt]

            if tokenizer.charMode:
                char = tokenizer.itos[ttt]
                print(char, end="", flush=True)
            else:
                char = tokenizer.tokenizer.decode(ctx[out_last:])
                if '\ufffd' not in char: # is valid utf8 string?
                    print(char, end="", flush=True)
                    out_last = i+1

        record_time('total')
        print(
            f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end = ''
        )

    print(("-" * 50) + '\n')
