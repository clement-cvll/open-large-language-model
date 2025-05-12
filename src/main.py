from model import Model
from config import Config
from transformers import AutoTokenizer
from trainer import load_best_checkpoint

# Initialize model and config
config = Config(batch_size=1)
model = Model(config)
model.to(config.device)
tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

# Load the best checkpoint
model = load_best_checkpoint(model, "best_checkpoint.pt")

# Test the model
model.eval()

# Generate text
input = "Hi, how are you"
input_ids = tokenizer.encode(input, return_tensors="pt").to(config.device)
text = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(text[0]))