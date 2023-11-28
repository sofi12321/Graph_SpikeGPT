import gdown
import json
from src.data.graph_dataset import *
from src.data.load_spikegpt_embedding import *

if __name__ == "__main__":
    # Clone SpikeGPT to the src.third_party folder
    args = prepare_env()

    emb = load_embedding_weights("/scr/third_party/SpikeGPT-OpenWebText-216M/", args)
    tokenizer = load_tokenizer()

    url = "https://drive.google.com/drive/folders/1ebERGfOqS_UrEHQG0Q4L8M266R9h1aPN?usp=drive_link"
    output = './'

    gdown.download_folder(url, output=output, quiet=True)

    # Load data
    file_dir = "./webnlg/"

    with open(file_dir + "train.json", "r") as f:
        train_set = json.load(f)

    with open(file_dir + "val.json", "r") as f:
        val_set = json.load(f)

    with open(file_dir + "test.json", "r") as f:
        test_set = json.load(f)


    # data = create_graph(get_triplets(test_set[0]), test_set[0]['text'][0], with_text=True)
    train_dataset = GraphDataset("./", "/data/processed/train.pt")
    val_dataset = GraphDataset("./", "/data/processed/val.pt")
    test_dataset = GraphDataset("./", "/data/processed/test.pt")