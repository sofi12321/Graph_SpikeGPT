import torch_geometric

def train(model, loss_fn, optimizer, data: torch_geometric.data.data.Data):
    """
    Training of one epoch

    :param model: GNN model
    :param loss_fn: loss function
    :param optimizer: optimizer
    :param data: torch_geometric dataset
    :return: current loss and output 
    """
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, out
