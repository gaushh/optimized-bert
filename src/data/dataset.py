from datasets import load_dataset
import yaml
import os

with open("../config/exp02_config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

if not os.path.isdir('../../data'):
  os.mkdir('../../data')

if not os.path.isdir('../../data/raw'):
  os.mkdir('../../data/raw')

if not os.path.isdir('../../data/processed'):
  os.mkdir('../../data/processed')

def dataset_to_text(dataset, output_filename="data.txt"):
  """Utility function to save dataset text to disk,
  useful for using the texts to train the tokenizer
  (as the tokenizer accepts files)"""
  with open(output_filename, "w") as f:
    for t in dataset["text"]:
      print(t, file=f)


dataset_bookcorpus_train = load_dataset("bookcorpus")["train"]
# print("dataset_bookcorpus_train : ", dataset_bookcorpus_train)
dataset_sampled = dataset_bookcorpus_train.train_test_split(test_size=config['dataset']["data_proportion"], seed=config["seed"])["test"]
# print("dataset_sampled : ", dataset_sampled)
dataset = dataset_sampled.train_test_split(test_size=config['dataset']["test_size"], seed=config["seed"])
# print("dataset3 : ", dataset)

# print(config['dataset']["raw_train_path"])
# if not os.path.isdir(config['dataset']["raw_train_path"]):
#   os.mkdir(config['dataset']["raw_train_path"])
#
# if not os.path.isdir(config['dataset']["raw_test_path"]):
#   os.mkdir(config['dataset']["raw_test_path"])

# save the training set to train.txt
dataset_to_text(dataset["train"], config['dataset']["raw_train_path"])
# save the testing set to test.txt
dataset_to_text(dataset["test"], config['dataset']["raw_test_path"])

dataset.save_to_disk(config['dataset']["raw_dataset_dir"])

