import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.load_spikegpt_embedding import *
from src.data.build_dataset import *
from src.data.graph_dataset import *

from src.model.graph2seq import *
from src.model.load_spikegpt_model import *
from src.vizualization.loss_plot import *

import warnings

warnings.simplefilter('ignore')


def predict(model: nn.Module,
            graph,
            criterion: nn.Module, tokenizer, device):
    # Make a single prediction
    model.eval()

    with torch.no_grad():
        graphs = graph.to(device)

        # training step for single batch
        labels = tokenize(graph.y.lower(), tokenizer)
        labels = F.pad(torch.tensor(labels), (1, 0), value=PAD_IDX).to(device)
        output = model(graphs.x, graphs.edge_index, labels.shape[0], 0)

        output = output[1:].view(-1, output.shape[-1])
        trg = labels[1:]

        loss = criterion(output, trg)
        cur_loss = loss.item()
        print("Loss:", cur_loss)

    return output.max(1)[1]


def run_graph_spike_gpt(graph, spike_gpt, args, graph_to_tokens, criterion, tokenizer, device, num_trials=2,
                        length_per_trial=100):
    # Run full model
    # GraphSpikeGPT
    context = graph.y
    ctx = predict(graph_to_tokens, graph, criterion, tokenizer, device).tolist()
    run_spike_gpt(spike_gpt, context, ctx, args, tokenizer, num_trials, length_per_trial)


if __name__ == "__main__":
    train_dataset = GraphDataset("./", "/data/processed/train.pt")
    test_dataset = GraphDataset("./", "/data/processed/test.pt")

    INPUT_DIM = train_dataset.num_features
    OUTPUT_DIM = 50277

    ENC_EMB_DIM = 768
    DEC_EMB_DIM = 768
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ATTN_DIM = 64
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph_to_tokens = Graph2Seq(enc, dec, device).to(device)

    args = prepare_env()
    tokenizer = load_tokenizer()
    spike_gpt = get_spike_gpt_model(args)

    # Load model weights
    version = 2
    torch.save(graph_to_tokens.state_dict(), )
    graph_to_tokens.load_state_dict(torch.load(f"/models/model_graph_to_tokens_{version}.pt", map_location=device))

    PAD_IDX = 1
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Run full model
    g_id = 2
    run_graph_spike_gpt(test_dataset[g_id], spike_gpt, args, graph_to_tokens, criterion, tokenizer, device)
