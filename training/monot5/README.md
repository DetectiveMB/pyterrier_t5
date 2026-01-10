## Training monoT5

Here we provide code we have used for training monoT5 models.

 - t5train.py - this uses the MSMARCO training triples for training monoT5. It also conducts validation.

 - t5-train-bm25negs.py - this uses the MSMARCO training queries for training monoT5. It adds negative samples obtained using BM25 from a Pisa index.

## Reproducibility of MonoT5

To replicate the performance of MonoT5 base on MSMARCO from the original monoT5 paper [1], you should use t5train.py with the following configuration: 

- train the model for one epoch on the small set of [MSMARCO-passage triples](https://huggingface.co/datasets/irds/msmarco-passage_train_triples-small)
- Since Nogueira et al. [1] limited the number of triples to 6.4e-5, number_df = 640000, which corresponds to 10k steps
- Use Adafactor as optimizer, with learning rate equal to 3e-4 and weigth decay equal to 5e-5
- Use batch size equal to 128 by applying gradient accumulation (mini batch size = 8, number of accumulation = 16)
- Truncate the input sentence with 'longest_first' option and max_length = 512

In this way, you can reproduce [MonoT5-base-10k](https://huggingface.co/castorini/monot5-base-msmarco-10k) with the following results:

|  **Model** | **AP** | **RR** | **nDCG@10** |
|:----------:|:------:|:------:|:-----------:|
| MonoT5 [1] | 0.368  |  0.947 |    0.699    |
| Replicated | 0.367  |  0.944 |    0.700    |


[1] [NOGUEIRA, Rodrigo, et al. Document Ranking with a Pretrained Sequence-to-Sequence Model. In: EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.63/)

## How to reproduce Light-MonoT5 [2]

To replicate the performance of Light-MonoT5 base on MS MARCO, you should use t5train.py with the same configuration as MonoT5 but with two differences: 

- Put number_df = 6400000, in this way we can compare with [Monot5-base-100k](https://huggingface.co/castorini/monot5-base-msmarco), which has been trained for 100k steps (and not 10k)
- The only parameters updated during the training are the embeddings of the prompt tokens, i.e. `Query`, `Document`, `Relevant`, `true`, `false`, `EoS` (`End of Sentence` token) and `Colon` (`:`). You need to add the following code to t5train.py:

````
# Register hook to zero out gradients for all tokens not in the prompt
def mask_gradients(grad):
    mask = torch.zeros_like(grad)
    # If you want to train the token Query
    mask[27569] = 1.0
    mask[3] = 1.0
    # If you want to train the token Document
    mask[11167] = 1.0
    # If you want to train the token Relevant
    mask[31484] = 1.0
    # If you want to train the tokens true and false
    mask[1176] = 1.0
    mask[6136] = 1.0
    # If you want to train the EoS token
    mask[1] = 1.0
    # If you want to train the Colon (:)
    mask[10] = 1.0
    
    return grad * mask

embedding_weight = model.get_input_embeddings().weight
embedding_weight.requires_grad = True
embedding_weight.register_hook(mask_gradients)
````

All the other parameters of the model must have `requires_grad = False`.

[2] Braga et al., 'Revealing MonoT5’s Learning Mechanisms via Prompt-Token Adaptation' in ECIR 2026

# Credits

Sean MacAvaney, University of Glasgow
Marco Braga, University of Milano-Bicocca
