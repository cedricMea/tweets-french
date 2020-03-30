import config
import torch

class BertDataset:

    def __init__(self, comments, targets):
        self.comments = comments
        self.targets = targets
        self.curr_tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):

        comment = self.comments[item]
        comment = " ".join(comment.split())  # To delete multiple space

        target = self.targets[item]
        inputs = self.curr_tokenizer.encode_plus(
            text=comment,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            pad_to_max_length=True
        )

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        # Attention_mask : mets a 0 les padding ajoutés
        # et 1 pour les vrais mots
        # special_tokens_masks : 1 pour les tokens speciaux (inclus CLS et SEP)

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_masks": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float)
        }
