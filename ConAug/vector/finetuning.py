from modelling import *
import argparse
import os
from torch import cuda
import pandas as pd

parser = argparse.ArgumentParser(description="Finetuning hyperparameters")
parser.add_argument('--batch_size', '-b', type=int, default=4, required=False, help="Train batch size")
parser.add_argument('--num_epochs', '-e', type=int, default=3, required=False, help="Number of epochs")
parser.add_argument('--learning_rate', '-l', type=float, default=1e-4, required=False, help="Learning rate")
parser.add_argument('--input_file', '-i', type=str, required=True, help="Input .h5 file (input_data_*)")
parser.add_argument('--output_dir', '-o', type=str, required=True, help="Output directory")
args = parser.parse_args()

# Setting up the device for GPU usage
device = 'cuda:1' if cuda.is_available() else 'cpu'


# NOTE
df = pd.read_hdf(args.input_file)

# let's define model parameters specific to T5
model_params = {
    "MODEL": "google/mt5-small",  # model_type: t5-base/t5-large TODO
    "TRAIN_BATCH_SIZE": args.batch_size,  # training batch size
    "VALID_BATCH_SIZE": args.batch_size,  # validation batch size
    "TRAIN_EPOCHS": args.num_epochs,  # number of training epochs TODO
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": args.learning_rate,  # learning rate TODO
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text TODO
    "MAX_TARGET_TEXT_LENGTH": 512,  # max length of target text TODO
    "SEED": 42,  # set seed for reproducibility
}


T5Trainer(
    dataframe=df,  # data
    source_text= "input",
    target_text="output",
    device=device,
    model_params=model_params,
    output_dir=args.output_dir,
)

