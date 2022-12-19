from transformers import BertTokenizerFast
from datasets import load_from_disk, concatenate_datasets
from itertools import chain
import yaml

with open("../config/exp02_config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

max_length = config["tokenizer"]["max_length"]
tokenizer_path = config["tokenizer"]["tokenizer_path"]
raw_dataset_dir = config["dataset"]["raw_dataset_dir"]
processed_train_dir = config['dataset']["processed_train_dir"]
processed_test_dir = config['dataset']["processed_test_dir"]
truncate_longer_samples = config["tokenizer"]["truncate_longer_samples"]



tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

def encode_with_truncation(examples):
  """Mapping function to tokenize the sentences passed with truncation"""
  return tokenizer(examples["text"], truncation=True, padding="max_length",
                   max_length=max_length, return_special_tokens_mask=True)

def encode_without_truncation(examples):
  """Mapping function to tokenize the sentences passed without truncation"""
  return tokenizer(examples["text"], return_special_tokens_mask=True)

# the encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation


dataset = load_from_disk(raw_dataset_dir)
# tokenizing the train dataset
train_dataset = dataset["train"].map(encode, batched=True)
# tokenizing the testing dataset
test_dataset = dataset["test"].map(encode, batched=True)


if truncate_longer_samples:
  # remove other columns and set input_ids and attention_mask as PyTorch tensors
  train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
  test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
  # remove other columns, and remain them as Python lists
  train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
  test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

if not truncate_longer_samples:
    train_dataset = train_dataset.map(group_texts, batched=True,
                                    desc=f"Grouping train texts in chunks of {max_length}")
    test_dataset = test_dataset.map(group_texts, batched=True,
                                  desc=f"Grouping test texts in chunks of {max_length}")
    # convert them from lists to torch tensors

train_dataset.set_format("torch")
test_dataset.set_format("torch")
train_dataset.save_to_disk(processed_train_dir)
test_dataset.save_to_disk(processed_test_dir)

