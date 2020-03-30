import pandas as pd
import datasets
import config 
import torch

def load_data(path):
    data = pd.read_csv(path, sep=";:!", engine="python") #  because I want all the data in one column

    # Create columns of interrest
    data["polarity"] = data[data.columns[0]].apply(lambda x: x[0])
    data["text"] = data[data.columns[0]].apply(lambda x: x[2:])
    data = data[["polarity", "text"]]

    # Delete wired lines pas beaucoup de lines dans la base finale
    data = data[data.polarity.isin(["0","4"])]

    # Supprimer les lignes vides
    data["Nb_mots"] = data["text"].apply(lambda x : len(x.split(" ")))
    data = data[(data.Nb_mots!=0) & data.text.str.len()!=0]

    # Conversion du label en int et 1 pour les comments positifs 
    data["polarity"] = data["polarity"].astype("int")
    data["polarity"] = data["polarity"].apply(lambda x: 1 if x==4 else x)

    return data[["polarity", "text"]]


def data_exploration(data):

    print("Different polarity and number of instances")
    print(f"{data.polarity.value_counts()}")
    data["Nb mots"] = data["text"].apply(lambda x : len(x.split(" ")))

    print("Texts lengths")
    print(data["Nb mots"].describe())



def create_data_loader(pandas_dataframe, is_train=True):
    """
    This function aims to create dataLoader for pytocrch training

    :pandas_dataframe : a pandas dataframe with columns ["text", "polarity"]
    """
    pandas_dataframe = pandas_dataframe.reset_index(drop=True)
    dataset_object = datasets.BertDataset(
        comments=pandas_dataframe.text.values,
        targets=pandas_dataframe.polarity.values
    )

    data_loader_object = torch.utils.data.DataLoader(
        dataset=dataset_object,
        batch_size= config.TRAIN_BATCH_SIZE if is_train else config.VALID_BATCH_SIZE,
        shuffle=True
    )

    return data_loader_object

