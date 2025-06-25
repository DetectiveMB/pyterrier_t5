## Training monoT5

Here we provide code we have used for training monoT5 models.

 - t5train.py - this uses the MSMARCO training triples for training monoT5. It also conducts validation.

 - t5-train-bm25negs.py - this uses the MSMARCO training queries for training monoT5. It adds negative samples obtained using BM25 from a Pisa index.

## Reproducibility of monoT5

To replicate the performance of monoT5 base on MSMARCO from the original monoT5 paper [1], you should use t5train.py with the following configuration: 

- train the model for one epoch on the small set of [MSMARCO-passage triples](https://huggingface.co/datasets/irds/msmarco-passage_train_triples-small)
- Since Nogueira et al. [1] limited the number of triples to 6.4e-5, number_df = 640000
- Use Adafactor as optimizer, with learning rate equal to 3e-4 and weigth decay equal to 5e-5
- Use batch size equal to 128 by applying gradient accumulation (mini batch size = 8, number of accumulation = 16)
- Truncate the input sentence with 'longest_first' option and max_length = 512

In this way, you can reproduce [monoT5-base-10k](https://huggingface.co/castorini/monot5-base-msmarco-10k) with the following results:

|  **Model** | **AP** | **RR** | **nDCG@10** |
|:----------:|:------:|:------:|:-----------:|
| monoT5 (1) | 0.368  |  0.947 |    0.699    |
| Replicated | 0.367  |  0.944 |    0.700    |


[1] [NOGUEIRA, Rodrigo, et al. Document Ranking with a Pretrained Sequence-to-Sequence Model. In: EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.63/)

# Credits

Sean MacAvaney, University of Glasgow
