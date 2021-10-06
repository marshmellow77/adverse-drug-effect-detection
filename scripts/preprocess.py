import transformers
from transformers import AutoTokenizer
import datasets
import pandas as pd
import numpy as np
import argparse
import os
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--datasubset-name", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))
    
    dataset = load_dataset(args.dataset_name, args.datasubset_name)
    df = pd.DataFrame(dataset['train'])
    print(f"Number of records before deduplication: {len(df)}")
    df = df.drop_duplicates()
    print(f"Number of records after deduplication: {len(df)}")
    
    # split the dataset into train, val, and test
    train, val, test = np.split(df.sample(frac=1), [int(args.train_ratio*len(df)), int((args.train_ratio+args.val_ratio)*len(df))])
    
    # convert the pandas dataframes into transformer datasets
    # more info on https://huggingface.co/docs/datasets/loading_datasets.html#from-a-pandas-dataframe
    train_ds = datasets.Dataset.from_pandas(train)   
    val_ds = datasets.Dataset.from_pandas(val)
    test_ds = datasets.Dataset.from_pandas(test)
    
    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)
    
    train_ds = train_ds.map(tokenize, batched=True, batch_size=len(train_ds))
    val_ds = val_ds.map(tokenize, batched=True, batch_size=len(val_ds))
    test_ds = test_ds.map(tokenize, batched=True, batch_size=len(test_ds))

    # upload data to S3
    train_ds.save_to_disk('/opt/ml/processing/training/')
    val_ds.save_to_disk('/opt/ml/processing/validation/')
    test_ds.save_to_disk('/opt/ml/processing/test/')
