### eventually, stuff will go here

from utilities.intermediate_decoding_gpt2 import GPT2Helper
from utilities.intermediate_decoding_llama2 import Llama7BHelper
from time import time
# token = ""
# gpt2_helper = GPT2Helper(token)

# prompt = "The capital of Germany is"

# generate_text = gpt2_helper.generate_text(prompt=prompt, max_length=25)
# print(generate_text)
token = "token"
model = Llama7BHelper(token)

print_attn_mech = False
print_intermediate_res = False
print_mlp = False
print_block = True

#"(b) Facial nerve repair(Ref. Scott Brown, 6th ed., 1404)Since generally following trauma the facial nerve injury occurs as sudden onset. Facial decompression should be the best option."

prompt = """<QUESTION>Treatment of choice in traumatic facial nerve injury is:
 (1) Facial sling
 (2) Facial nerve repair
 (3) Conservative management
 (4) Systemic corticosteroids</QUESTION>
<ANSWER>"""
prompt2 = "The most important political question in the world is"
# model.decode_all_layers(prompt, 
#                         print_attn_mech = print_attn_mech,
#                         print_intermediate_res = print_intermediate_res, 
#                         print_mlp = print_mlp, 
#                         print_block = print_block)
time_before = time()
text = model.generate_text(prompt, max_new_tokens=10)
time_after = time()

print(text)

num_new_tokens = model.count_new_tokens(prompt, text)
time_for_generation = time_after - time_before
tokens_per_second = num_new_tokens / time_for_generation
print(f"Time taken: {time_for_generation}")
print(f"Tokens per second: {tokens_per_second}")
