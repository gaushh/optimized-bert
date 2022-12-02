import os
import yaml
from datasets import load_from_disk
from transformers import Trainer
from transformers import BertTokenizerFast
from transformers import TrainingArguments
from transformers import BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling

with open("../config/config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

print("Hello 1")

train_path = config['dataset']['processed_train_dir']
test_path = config['dataset']['processed_test_dir']

train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

print("Hello 2")

vocab_size = config['tokenizer']["vocab_size"]
max_length = config['tokenizer']["max_length"]
tokenizer_path = config["tokenizer"]["tokenizer_path"]

evaluation_strategy = config['bert']["evaluation_strategy"]
mlm_probability = config['bert']["mlm_probability"]
model_path = config['bert']["model_path"]
num_train_epochs = config['bert']["num_train_epochs"]
per_device_train_batch_size = config['bert']["per_device_train_batch_size"]
per_device_eval_batch_size = config['bert']["per_device_eval_batch_size"]
gradient_accumulation_steps = config['bert']["gradient_accumulation_steps"]
logging_steps = config['bert']["logging_steps"]
save_steps = config['bert']["save_steps"]

print("Hello 3")

tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

print("Hello 4")

model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)

print("Hello 5")


if not os.path.isdir(model_path):
  os.mkdir(model_path)


training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy=evaluation_strategy,    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,            # number of training epochs, feel free to tweak
    per_device_train_batch_size=per_device_train_batch_size, # the training batch size, put it as high as your GPU memory fits
    per_device_eval_batch_size=per_device_eval_batch_size,  # evaluation batch size
    gradient_accumulation_steps=gradient_accumulation_steps,  # accumulating the gradients before updating the weights
    logging_steps=logging_steps,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=save_steps,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
    # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
)


print("Hello 6")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=test_dataset,
    eval_dataset=test_dataset,
)

print("Hello 7")

trainer.train()

print("Hello 8")