from transformers import CamembertModel
from transformers import AdamW

if __name__=="__main__":
    model = CamembertModel.from_pretrained("camembert-base")

    param_optimizer = list(model.named_parameters())
    no_decay = ["biais", "LayerNorm.biais", "LayerNorm.weight"]

    optimizer_parameters = [
        {'params': [tensor for name, tensor in param_optimizer if name in no_decay], 'weight_decay': 0},
        {'params': [tensor for name, tensor in param_optimizer if name not in no_decay], 'weight_decay':0.01}
    ] 

    optimizer_parameters_him = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    # This is the overall number of trainings that will be performed
    num_training_steps = 50
    optimizer = AdamW(optimizer_parameters, lr=3*10-5)  # Arbitrary set

