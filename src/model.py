# import transformers
import torch.nn as nn
import config


class BuildModel(nn.Module):
    def __init__(self):
        super(BuildModel, self).__init__()

        # Define different layers for the model
        self.bert_layer = config.CAMEMBERT_MODEL
        self.dropout_layer = nn.Dropout(0.3)
        self.dense_layer = nn.Linear(768, 1)
        # 768 because in default BretConfig Hidden_size=768,
        #    1 because it is binary classif

    def forward(self, inputs_id, mask, sentence_id):
        # out_sentences, out_pooling
        _, out_pooling = self.bert_layer(
            inputs_id,
            attention_mask=mask,
            token_type_ids=sentence_id
        )

        # Output_pooling is a vector of [1, 768] represents the whole sentence.
        #   and is use for sentence classification
        # Output_sentence is a tensor of [MAX_LEN, 768]. It contains the hidden
        #   layer of each word of the sentence
        # When Using TensorFlow for Sequence Classification,
        #   out_pooling is at the first place
        dropout = self.dropout_layer(out_pooling)
        output = self.dense_layer(dropout)

        return output
