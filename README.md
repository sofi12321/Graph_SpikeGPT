# Graph_SpikeGPT
Named Entity Graph Recognition via SpikeGPT

The essence of our task is to generate text by Knowledge Graph (KG) with the Spike-based Transformer. We conducted a literature review and found articles [1, 2] dedicated to the realization of our hypothesis. 
Li et al.[1] generated meaningful natural language text from a knowledge graph, where entities are represented as nodes and links between entities as edges. 
On the other hand, Zhu et al. [2] changed the Transformer structure with main blocks as spike-based ones. Combining both approaches together we plan to perform text generation given KG with Spike-based Transformer.
In addition, the analysis of datasets was performed and it was decided to use less complex datasets like WebNLG or GenWiki for initial tests.

<p align="center">
  <img height="270" src="https://github.com/sofi12321/Graph_SpikeGPT/blob/main/static/graph_example.png">
</p>

Our proposed model consists of two parts:
1. GNN. The input data of the model is Knowledge Graph (in particular represented in torch_geometric.data.data.Data format). However, in the original SpikeGPT model [2], there is a block of text embedding at the very beginning. We run the text corresponding to the graph through the pre-trained text embedding block, which results in a vector of features. To be able to process the graph, we propose to train a GNN using the graph as input and the vector of features obtained from the text embedding block as targets. Then in use, the GNN is substituted instead of the text embedding block.

<p align="center">
  <img height="720" src="https://github.com/sofi12321/Graph_SpikeGPT/blob/main/static/Graph_SpikeGPT.png">
</p>

2. SpikeGPT pre-trained model. According to [2] SpikeGPT adopts the Transformer paradigm of alternating token-mixer and channel-mixer layers. In SpikeGPT, the token-mixer is a Spiking RWKV layer, which replaces the self-attention mechanism with an RNN-like structure that preserves parallelization. To maintain consistency with the binary activations of SNNs, a binary embedding step is proposed to convert the continuous outputs of the embedding layer into binary spikes. The token shift operator combines information from the global context with information of the original token to provide the token with better contextual information.

<p align="center">
  <img height="500" src="https://github.com/sofi12321/Graph_SpikeGPT/blob/main/static/addition.png">
</p>

Reference:

[1]
J. Li, T. Tang, W. X. Zhao, Z. Wei, N. J. Yuan, and J.-R. Wen, “Few-shot knowledge graph-to-text generation with pretrained language models,” in Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, 2021.

[2]
R.-J. Zhu, Q. Zhao, G. Li, and J. K. Eshraghian, “SpikeGPT: Generative pre-trained language model with spiking Neural Networks,” arXiv [cs.CL], 2023.


