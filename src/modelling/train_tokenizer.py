import os
import json
import yaml
from tokenizers import BertWordPieceTokenizer



with open("../config/config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


files = config['tokenizer']["files"]
max_length = config['tokenizer']["max_length"]
vocab_size = config['tokenizer']["vocab_size"]
special_tokens = config['tokenizer']["special_tokens"]
tokenizer_path = config["tokenizer"]["tokenizer_path"]

# initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer()

print(files)
print(vocab_size)
print(special_tokens)

# train the tokenizer
tokenizer.train(files= files, vocab_size= vocab_size, special_tokens= special_tokens)

# enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length)


if not os.path.isdir(tokenizer_path):
  os.mkdir(tokenizer_path)

# save the tokenizer
tokenizer.save_model(tokenizer_path)

# dumping some of the tokenizer config to config file,
# including special tokens, whether to lower case and the maximum sequence length
with open(os.path.join(tokenizer_path, "config.json"), "w") as f:
    tokenizer_cfg = {
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "model_max_length": max_length,
        "max_len": max_length
    }
    json.dump(tokenizer_cfg, f)





