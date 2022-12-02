from datasets import load_dataset
import yaml
import os

with open("../config/config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def dataset_to_text(dataset, output_filename="data.txt"):
  """Utility function to save dataset text to disk,
  useful for using the texts to train the tokenizer
  (as the tokenizer accepts files)"""
  with open(output_filename, "w") as f:
    for t in dataset["text"]:
      print(t, file=f)


dataset = load_dataset("bookcorpus")["train"]
dataset = dataset.train_test_split(config['dataset']["test_size"], config["seed"])

# save the training set to train.txt
dataset_to_text(dataset["train"], config['dataset']["raw_train_path"])
# save the testing set to test.txt
dataset_to_text(dataset["test"], config['dataset']["raw_test_path"])

dataset.save_to_disk(config['dataset']["raw_dataset_dir"])

