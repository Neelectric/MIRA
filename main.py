### eventually, stuff will go here

from utilities.intermediate_decoding import GPT2Helper

gpt2_helper = GPT2Helper

prompt = "The capital of Germany is"

generate_text = gpt2_helper.generate_text(prompt, max_length=25)