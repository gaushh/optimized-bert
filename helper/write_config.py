import yaml
import os

config_info = {'seed': 43,
               'dataset': {
                   'test_size': 0.01,
                   'raw_train_path': "../../data/raw/train.txt",
                   'raw_test_path': "../../data/raw/test.txt",
                   'raw_dataset_dir': "../../data/raw/raw_dataset",
                   'processed_train_dir': "../../data/processed/processed_train",
                   'processed_test_dir': "../../data/processed/processed_test"
               },
               'tokenizer': {
                   'vocab_size': 30_522,
                   'max_length': 512,
                   'truncate_longer_samples': False,
                   'files': ["../../data/raw/test.txt"],
                   'tokenizer_path': "../../models/tokenizer",
                   'special_tokens': ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
               },
               'bert': {
                   'model_path': "../../models/exp01",
                   'mlm_probability': 0.15,
                   'evaluation_strategy': "steps",
                   'num_train_epochs': 1,
                   'per_device_train_batch_size': 10,
                   'gradient_accumulation_steps': 8,
                   'per_device_eval_batch_size': 64,
                   'logging_steps': 100,
                   'save_steps': 100,
               }
               }


with open("../src/config/config.yaml", 'w') as yamlfile:
    data = yaml.dump(config_info, yamlfile)
    print("Write successful")