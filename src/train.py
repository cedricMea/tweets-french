import numpy as np 
from sklearn import model_selection, metrics
import torch
from  model import BuildModel
from transformers import AdamW, get_linear_schedule_with_warmup
import config
import utils
import engine

def run():
    
    # Load data and little exploration
    data = utils.load_data(config.DATA_PATH)
    utils.data_exploration(data)  # Data exploration just print
    
    # print(data.head())
    # print(data.polarity.values)
    # Create dataLoader

    data, _ = model_selection.train_test_split(
        data,
        test_size=0.995,
        random_state=42,
        stratify=data.polarity.values
    )

    train, valid = model_selection.train_test_split(
        data,
        test_size=0.5,
        random_state=42,
        stratify=data.polarity.values
    )

    train_data_loader = utils.create_data_loader(train)
    valid_data_loader = utils.create_data_loader(valid, is_train=False)

    #S

    # Build Model and send it to device 
    model = BuildModel()
    model = model.to(config.DEVICE)

    # Set weight decay to 0 fro no_decay params
    # Set weights decays to 0.01 for others
    param_optimizer = list(model.named_parameters())
    no_decay = ["biais", "LayerNorm.biais", "LayerNorm.weight"]

    optimizer_parameters = [
        {'params': [tensor for name, tensor in param_optimizer if name in no_decay], 'weight_decay': 0},
        {'params': [tensor for name, tensor in param_optimizer if name not in no_decay], 'weight_decay':config.WEIGHT_DECAY}
    ] 

    # This is the overall number of trainings that will be performed
    num_training_steps = int((train.shape[0] / config.TRAIN_BATCH_SIZE) * config.EPOCHS)

    optimizer = AdamW(optimizer_parameters, lr=3*10-5)  # Arbitrary set

    # Scheduler to performs adaptatite LR regarding epochs number
    # Scheduler_with_warm_up consiste a augmenter le LR dans les premiers 
    # warm_up steps afin de converger plus vite dans les debuts
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=4
    )


    # model = nn.DataParallel(model) # if multiples GPU

    best_accuracy = 0
    best_model_state = None

    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model)
        outputs = np.where(np.array(outputs)>0.5, 1, 0)
        accuracy = metrics.accuracy_score(np.array(targets), outputs)
        print(f"Accuracy, Epoch {epoch} : {accuracy}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict 
    
    print("Best accuracy : {best_accuracy}" )
    
    print("Saving Model...")
    torch.save(best_model_state, config.MODEL_PATH)










    


if __name__=="__main__":
    run()
