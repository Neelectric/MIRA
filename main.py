### eventually, stuff will go here

from utilities.intermediate_decoding_gpt2 import GPT2Helper
token = ""
gpt2_helper = GPT2Helper(token)

prompt = "The capital of Germany is"

generate_text = gpt2_helper.generate_text(prompt=prompt, max_length=25)
print(generate_text)