from transformers import CamembertTokenizer, CamembertModel
import torch
import os as os
import urllib3

# This env variable will disable SSL verification
# for requests (because of Milliman SSL)
# https://stackoverflow.com/questions/48391750/disable-python-requests-ssl-validation-for-an-imported-module
os.environ["CURL_CA_BUNDLE"] = ""
# disable warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")



BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
DATA_PATH = os.path.join(BASE_PATH, "input/tweets.csv")
MODEL_PATH = os.path.join(BASE_PATH, "output/camembert")  # MODEL SAVE PATH


# Set Pytorch device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EPOCHS = 10
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4 
WEIGHT_DECAY = 0.01
TOKENIZER = CamembertTokenizer.from_pretrained(
    "camembert-base",
    do_lower_case=True,
    verify=False,
)
CAMEMBERT_MODEL = CamembertModel.from_pretrained("camembert-base")

# ? valid batch size different de train batch size
# ? BCEWithLogit more than BCE

