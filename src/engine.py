from tqdm import tqdm
import torch
import config


def loss_fn(predict, real_values):
    # BCE binary cross entropy
    return torch.nn.BCEWithLogitsLoss()(predict, real_values.view([-1,1]))


def train_fn(data_loader, model, optimizer, scheduler):
    # Declare the state of your model
    # Because all layers doesnt have the same behaviour 
    # depending the state (dropout, BatchNorm)
    model.train()

    for batch_index, batch_dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
        # Get diffents input from batch dataset
        batch_ids = batch_dataset["ids"]
        batch_attention_masks = batch_dataset["attention_masks"]
        batch_inputs_tokens_ids = batch_dataset["token_type_ids"]
        batch_targets = batch_dataset["target"]

        # Move these differents inputs to GPU
        batch_ids = batch_ids.to(config.DEVICE, dtype=torch.long)
        batch_attention_masks = batch_attention_masks.to(config.DEVICE, dtype=torch.long)
        batch_inputs_tokens_ids = batch_inputs_tokens_ids.to(config.DEVICE, dtype=torch.long)
        batch_targets = batch_targets.to(config.DEVICE, dtype=torch.float)

        optimizer.zero_grad()  # reset des gradients

        # Forward, Loss computation and backProp
        outputs = model(
            inputs_id=batch_ids,
            mask=batch_attention_masks,
            sentence_id=batch_inputs_tokens_ids
        )
        loss = loss_fn(outputs, batch_targets)
        loss.backward()  # Calcul des nouveaux gradients
        optimizer.step()  # Mise à jour des poids
        scheduler.step()
        # Scheduler permet d'avoir learning rate adaptatif
        # en fonction du nombre d'epochs



def eval_fn(data_loader, model):

    all_targets = []
    all_outputs = []
    # Eval will put layers like dropout or
    # BatchNorm in an evaluation mode
    model.eval()

    # torch.nograd will descativate gradient saving 
    # and then accelerate computing
    with torch.no_grad():
        for batch_index, batch_dataset in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Get diffents input from batch dataset
            batch_ids = batch_dataset["ids"]
            batch_attention_masks = batch_dataset["attention_masks"]
            batch_inputs_tokens_ids = batch_dataset["token_type_ids"]
            batch_targets = batch_dataset["target"]


            # Move these differents inputs to GPU
            batch_ids = batch_ids.to(config.DEVICE, dtype=torch.long)
            batch_attention_masks = batch_attention_masks.to(config.DEVICE, dtype=torch.long)
            batch_inputs_tokens_ids = batch_inputs_tokens_ids.to(config.DEVICE, dtype=torch.long)
            batch_targets = batch_targets.to(config.DEVICE, dtype=torch.float)


            batch_outputs = model(
                inputs_id=batch_ids,
                mask=batch_attention_masks,
                sentence_id=batch_inputs_tokens_ids
            )

            all_targets.extend(batch_targets.cpu().detach().numpy().tolist())  # Save all targets
            all_outputs.extend(torch.sigmoid(batch_outputs).cpu().detach().numpy().tolist()) # Apply sigmoids and save predictions


    return all_outputs, all_targets
