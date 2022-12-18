import os
import yaml
from datasets import load_from_disk
from transformers import Trainer
from transformers import BertTokenizerFast
from transformers import TrainingArguments
from transformers import BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
import wandb

with open("../config/config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

wandb.init(project="optimized-bert", entity="madridistas")

print("Loading Data ...")

train_path = config['dataset']['processed_train_dir']
test_path = config['dataset']['processed_test_dir']

train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

print("Reading hyperparameters from config ... ")

vocab_size = config['tokenizer']["vocab_size"]
max_length = config['tokenizer']["max_length"]
tokenizer_path = config["tokenizer"]["tokenizer_path"]

# output_dir = config['bert']["output_dir"]
model_path = config['bert']["model_path"]
evaluation_strategy = config['bert']["evaluation_strategy"]
mlm_probability = config['bert']["mlm_probability"]
num_train_epochs = config['bert']["num_train_epochs"]
# per_device_train_batch_size = config['bert']["per_device_train_batch_size"]
# per_device_eval_batch_size = config['bert']["per_device_eval_batch_size"]
gradient_accumulation_steps = config['bert']["gradient_accumulation_steps"]
logging_steps = config['bert']["logging_steps"]
save_steps = config['bert']["save_steps"]

learning_rate = config['bert']["learning_rate"]
weight_decay = config['bert']["weight_decay"]
adam_beta1 = config['bert']["adam_beta1"]
adam_beta2 = config['bert']["adam_beta2"]
lr_scheduler_type = config['bert']["lr_scheduler_type"]
warmup_ratio = config['bert']["warmup_ratio"]
auto_find_batch_size = config['bert']["auto_find_batch_size"]

push_to_hub = config['bert']["push_to_hub"]
hub_token = config['bert']["hub_token"]
hub_private_repo = config['bert']["hub_private_repo"]
hub_strategy = config['bert']["hub_strategy"]
hub_model_id = config['bert']["hub_model_id"]

print("Loading tokenizer and model ...")

tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)


model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability)



if not os.path.isdir(model_path):
  os.mkdir(model_path)


training_args = TrainingArguments(
    output_dir=model_path,          # output directory to where save model checkpoint
    evaluation_strategy=evaluation_strategy,    # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,            # number of training epochs, feel free to tweak
    # per_device_train_batch_size=per_device_train_batch_size, # the training batch size, put it as high as your GPU memory fits
    # per_device_eval_batch_size=per_device_eval_batch_size,  # evaluation batch size
    auto_find_batch_size = auto_find_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,  # accumulating the gradients before updating the weights
    logging_steps=logging_steps,             # evaluate, log and save model checkpoints every 1000 step
    save_steps=save_steps,
    report_to="wandb",
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    adam_beta1=adam_beta1,
    adam_beta2=adam_beta2,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
    push_to_hub=push_to_hub,
    hub_token=hub_token,
    hub_private_repo=hub_private_repo,
    hub_strategy=hub_strategy,
    hub_model_id=hub_model_id
    )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Starting Training ...")

trainer.train()

print("Finished")