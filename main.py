### eventually, stuff will go here

from utilities.intermediate_decoding_gpt2 import GPT2Helper
from utilities.intermediate_decoding_llama2 import Llama7BHelper
# token = ""
# gpt2_helper = GPT2Helper(token)

# prompt = "The capital of Germany is"

# generate_text = gpt2_helper.generate_text(prompt=prompt, max_length=25)
# print(generate_text)
token = "token"

model = Llama7BHelper(token)
model.decode_all_layers('The most important political question in the world is', print_intermediate_res=False, print_mlp=False,print_block=False)