import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import time

from src.data.load_spikegpt_embedding import *
from src.data.build_dataset import *
from src.data.graph_dataset import *

from src.model.graph2seq import *
from src.model.load_spikegpt_model import *

import warnings

warnings.simplefilter('ignore')


def train(model: nn.Module,
          train_dataset,
          optimizer: torch.optim.Optimizer,
          criterion: nn.Module,
          clip: float,
          batch_num, batch_size, tokenizer, device):
    # Training loop

    model.train()

    epoch_loss = 0
    counter = 0

    # progress bar
    progress = tqdm(range(batch_num * batch_size, min((batch_num + 1) * batch_size, len(train_dataset))),
                    desc="Train loss: ", total=batch_size)

    for i in progress:
        # training step for single batch
        labels = tokenize(train_dataset[i].y, tokenizer)
        labels = F.pad(torch.tensor(labels), (1, 0), value=PAD_IDX).to(device)

        graphs = train_dataset[i].to(device)
        output = model(graphs.x, graphs.edge_index, labels, 0.5)
        output = output[1:].view(-1, output.shape[-1])
        trg = labels[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        cur_loss = loss.item()
        epoch_loss += cur_loss
        counter += 1

        progress.set_description("Train loss: {:.7f}".format(epoch_loss / counter))

    if counter == 0:
        return 0
    return epoch_loss / batch_size


def evaluate(model: nn.Module,
             train_dataset,
             criterion: nn.Module, batch_num, batch_size, tokenizer, device):
    model.eval()
    counter = 0
    epoch_loss = 0

    with torch.no_grad():
        # progress bar
        progress = tqdm(range(batch_num * batch_size, min((batch_num + 1) * batch_size, len(train_dataset))),
                        desc="Validation loss:", total=batch_size)

        for i in progress:
            graphs = train_dataset[i].to(device)

            # training step for single batch
            labels = tokenize(train_dataset[i].y, tokenizer)
            labels = F.pad(torch.tensor(labels), (1, 0), value=PAD_IDX).to(device)
            output = model(graphs.x, graphs.edge_index, labels.shape[0], 0)

            output = output[1:].view(-1, output.shape[-1])
            trg = labels[1:].view(-1)
            loss = criterion(output, trg)

            cur_loss = loss.item()
            epoch_loss += cur_loss
            counter += 1

            progress.set_description("Validation loss: {:.7f}".format(epoch_loss / counter))

    if counter == 0:
        return 0
    return epoch_loss / counter


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def run_epochs(batch_size, val_size, test_size, train_batch_num, tokenizer, device, n_epochs=1):
    CLIP = 1
    batch_num = train_batch_num
    graph_to_tokens.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):

        start_time = time.time()
        for batch_id in range(batch_num):
            # Train on subset
            train_loss = train(graph_to_tokens, train_dataset, optimizer, criterion, CLIP, batch_id, batch_size,
                               tokenizer, device)
            # Evaluate on subset
            valid_loss = evaluate(graph_to_tokens, val_dataset, criterion, batch_id, val_size, tokenizer, device)

            train_losses.append(train_loss)
            val_losses.append(valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'\n\nEpoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    test_loss = evaluate(graph_to_tokens, test_dataset, criterion, 1, test_size, tokenizer, device)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    return train_losses, val_losses, test_loss


if __name__ == "__main__":
    # HIDDEN_DIM = 1024
    # OUTPUT_DIM = 50277
    # MAX_OUT_SIZE = 103

    train_dataset = GraphDataset("./", "/data/processed/train.pt")
    val_dataset = GraphDataset("./", "/data/processed/val.pt")

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


    def init_weights(m: nn.Module):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)


    graph_to_tokens.apply(init_weights)
    graph_to_tokens.decoder.embedding.weight.data = spike_gpt.w.emb.weight.data
    graph_to_tokens.decoder.embedding.requires_grad = False

    optimizer = torch.optim.Adam(graph_to_tokens.parameters())

    PAD_IDX = 1

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Run 2 epochs
    train_losses, val_losses, test_losses = run_epochs(15, 3, 15, 20, tokenizer, device, n_epochs=2)

    # Save model weights
    version = 3
    torch.save(graph_to_tokens.state_dict(), f"/models/model_graph_to_tokens_{version}.pt")
