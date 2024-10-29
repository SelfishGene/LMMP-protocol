#%% Imports

import os
import re
import glob
import time
import base64
import tiktoken
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import google.generativeai as genai
from lmmp import LargeMultimodalModelProtocol
from api_key_manager import initialize_api_keys, print_api_keys

#%% setup API keys

api_keys_dict = {
    'GEMINI_API_KEY': 'your_gemini_api_key_here'
}

initialize_api_keys(api_keys_dict)
print_api_keys()

#%% setup gemini model

GEMINI_MODEL_NAME = "gemini-1.5-flash-002"
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
gemini_model = genai.GenerativeModel(model_name=f"models/{GEMINI_MODEL_NAME}")

#%% load the LMMP model

current_file_path = os.path.abspath(__file__)
dictionary_folder = os.path.join(os.path.dirname(current_file_path), 'patch_dicts')
grid_image_path = os.path.join(dictionary_folder, 'patch_grid_p128_g64_8192x8192.jpg')

# Initialize LMMP
LMMP = LargeMultimodalModelProtocol(grid_image_path)

#%% gather ~100 images from the dataset and their corresponding textual descriptions

dataset_path = r"data/SFHQ_T2I_dataset"

csv_path = os.path.join(dataset_path, "SFHQ_T2I_dataset.csv")
image_folder = os.path.join(dataset_path, "images")

# Load the dataset csv file
df = pd.read_csv(csv_path)

num_images_to_use = 500

# sample 100 random images
df_to_use = df.sample(n=num_images_to_use)

image_paths_list = [os.path.join(image_folder, image_filename) for image_filename in df_to_use['image_filename']]
image_descriptions_list = df_to_use['text_prompt'].tolist()

patch_sizes = [32, 16]
patch_fractions = [1.0, 0.25]
patch_quantizations = ['rgb', 'rgb']

patch_sizes = [32]
patch_fractions = [1.0]
patch_quantizations = ['rgb']

patch_sizes = [64]
patch_fractions = [1.0]
patch_quantizations = ['rgb']

# go over all images and encode them
image_str_encodings_list = []
for image_path in image_paths_list:
    image_PIL = Image.open(image_path)

    encoded_str = LMMP.image2string(image_PIL, patch_sizes, patch_fractions, patch_quantizations)
    # image_PIL_rec = LMMP.string2image(encoded_str)

    image_str_encodings_list.append(encoded_str)

#%% assemble a prompt of several (image -> text) pairs

def format_prompt(prompt, max_width=85, min_width=55):
    words = prompt.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 > max_width:
            lines.append(current_line)
            current_line = word
        else:
            if current_line:
                current_line += " " + word
            else:
                current_line = word

            if len(current_line) >= min_width and (current_line.endswith(',') or current_line.endswith('.')):
                lines.append(current_line) 
                current_line = ""

    if current_line:
        lines.append(current_line)

    return '\n'.join(lines)

# split our 100 images into train and test sets
num_train_images = 200
num_test_images = len(image_paths_list) - num_train_images

train_images_paths_list = image_paths_list[:num_train_images]
train_image_str_encodings_list = image_str_encodings_list[:num_train_images]
train_image_descriptions_list = image_descriptions_list[:num_train_images]

test_images_paths_list = image_paths_list[num_train_images:]
test_image_str_encodings_list = image_str_encodings_list[num_train_images:]
test_image_descriptions_list = image_descriptions_list[num_train_images:]

# randomly select images to be in the prompt context
num_images_for_prompt_context = 100

# create a prompt with all the images and their descriptions
context_prompt = f''
for i in range(num_images_for_prompt_context):
    current_sample_prompt = f'\nimage string representation: \n{train_image_str_encodings_list[i]} \n\nimage description: \n{train_image_descriptions_list[i]}'
    context_prompt += current_sample_prompt

# assembe a query prompt
random_query_image_index = np.random.randint(len(test_image_descriptions_list))
query_image_str_encoding = test_image_str_encodings_list[random_query_image_index]
query_image_description = test_image_descriptions_list[random_query_image_index]

query_prompt = f'Given the above image string representations and their descriptions, generate a description for the following image, include hair, shirt and background dominant colors: \n\n'
query_prompt += f'image string representation: \n{query_image_str_encoding} \n\nimage description: \n'

print('------------------------------------------------')
print(f'lenght of context prompt: {len(context_prompt)}')
print(f'lenght of query prompt: {len(query_prompt)}')
print('------------------------------------------------')

# display the description of the context and query examples
for i in range(min(num_images_for_prompt_context, 5)):
    formatted_context_prompt = format_prompt(train_image_descriptions_list[i])
    print('------------------------------')
    print(f'context description {i}: \n{formatted_context_prompt}')
    print('------------------------------')

print('------------------------------------------------')
formatted_query_prompt = format_prompt(query_image_description)
print(f'query description: \n{formatted_query_prompt}')
print('------------------------------------------------')

#%% send the prompt to the gemini model

start_time = time.time()
input_content = [context_prompt, query_prompt]
response = gemini_model.generate_content(
    input_content,
    generation_config={"temperature": 0.7}
)
duration_sec = time.time() - start_time

prompt_token_count = response.usage_metadata.prompt_token_count
output_token_count = response.usage_metadata.candidates_token_count
output_description = response.text

print('------------------------------------------------')
print(f'prompt token count: {prompt_token_count}')
print(f'output token count: {output_token_count}')
print('------------------------------------------------')
print(f'ground truth description: \n{format_prompt(query_image_description)}')
print('------------------------------------------------')
print(f'gemini output description: \n{format_prompt(output_description)}')
print('------------------------------------------------')
print(f'gemini generation took: {duration_sec:.2f} seconds')
print('------------------------------------------------')


#%% visualize original and reconstructed images with descriptions

# Get the query image path and reconstruct the image from string representation
query_image_path = test_images_paths_list[random_query_image_index]
query_image = Image.open(query_image_path)
reconstructed_image = LMMP.string2image(query_image_str_encoding)

# display the orig and rec images side by side, along with descriptions of GT and gemini output
plt.figure(figsize=(20, 12))
plt.subplots_adjust(wspace=0.01, top=0.91)
plt.suptitle(f"gen time: {duration_sec:.2f}s, in tokens: {prompt_token_count}, out tokens: {output_token_count}", fontsize=14, y=1.02)

plt.subplot(1, 2, 1)
plt.imshow(query_image)
plt.title(f'Original Image Ground Truth Description:\n "{format_prompt(query_image_description)}"', fontsize=12, pad=10)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image)
plt.title(f'Reconstructed Image Geminis Description:\n "{format_prompt(output_description)}"', fontsize=12, pad=10)
plt.axis('off')

#%% setup text-to-image pairs similar to previous image-to-text pairs

# split our 100 images into train and test sets
num_train_images = 200
num_test_images = len(image_paths_list) - num_train_images

train_images_paths_list = image_paths_list[:num_train_images]
train_image_str_encodings_list = image_str_encodings_list[:num_train_images]
train_image_descriptions_list = image_descriptions_list[:num_train_images]

test_images_paths_list = image_paths_list[num_train_images:]
test_image_str_encodings_list = image_str_encodings_list[num_train_images:]
test_image_descriptions_list = image_descriptions_list[num_train_images:]

# assembe a query prompt
random_query_image_index = np.random.randint(len(test_image_descriptions_list))
query_image_str_encoding = test_image_str_encodings_list[random_query_image_index]
query_image_description = test_image_descriptions_list[random_query_image_index]

# randomly select images to be in the prompt context
num_images_for_prompt_context = 125

# Assemble context prompt with text -> image pairs
context_prompt_t2i = f''
for i in range(num_images_for_prompt_context):
    current_sample_prompt = f'\nimage description: \n{train_image_descriptions_list[i]} \n\nimage string representation: \n{train_image_str_encodings_list[i]}'
    context_prompt_t2i += current_sample_prompt

query_image_description = '''
Expressive portrait of a 66-year-old Latin American man with straight maroon hair and
ice blue eyes, looking up. olive skin tone, bewildered expression.
in sportswear, in candlelight, on a farm. with a goatee. 4K, Lifelike, Emotive.
'''

# Assemble query prompt for text-to-image generation
query_prompt_t2i = f'Given the above descriptions and their image string representations, generate an image string representation for the following description: \n\n'
query_prompt_t2i += f'image description: \n{query_image_description} \n\nimage string representation: \n'

print('------------------------------------------------')
print(f'length of context prompt: {len(context_prompt_t2i)}')
print(f'length of query prompt: {len(query_prompt_t2i)}')
print('------------------------------------------------')
print('------------------------------------------------')
print(f'query description: \n{format_prompt(query_image_description)}')
print('------------------------------------------------')

#%% send the text-to-image prompt to gemini model

start_time = time.time()
input_content = [context_prompt_t2i, query_prompt_t2i]
response_t2i = gemini_model.generate_content(
    input_content,
    generation_config={"temperature": 0.7}
)
duration_sec_t2i = time.time() - start_time

prompt_token_count_t2i = response_t2i.usage_metadata.prompt_token_count
output_token_count_t2i = response_t2i.usage_metadata.candidates_token_count
output_image_str = response_t2i.text

print('------------------------------------------------')
print(f'prompt token count: {prompt_token_count_t2i}')
print(f'output token count: {output_token_count_t2i}')
print('------------------------------------------------')
print(f'duration of gemini generation: {duration_sec_t2i:.2f} seconds')
print('------------------------------------------------')
print(f'gemini output image string: \n{output_image_str[:500]}')
print('...')
print('...')
print('...')
print(f'{output_image_str[-500:]}')
print('------------------------------------------------')

#%% visualize results in 2x2 grid

# Get the query image path and reconstruct images from string representations
query_image_path = test_images_paths_list[random_query_image_index]
query_image = Image.open(query_image_path)
reconstructed_image_gt = LMMP.string2image(query_image_str_encoding)
reconstructed_image_gemini = LMMP.string2image(output_image_str)

# Create 2x2 subplot
plt.figure(figsize=(20, 24))
plt.subplots_adjust(hspace=0.1, wspace=0.01)

# Add overall title with generation stats for both models
suptitle_text = (
    f"Image-to-Text: time={duration_sec:.2f}s, in={prompt_token_count}, out={output_token_count}\n" +
    f"Text-to-Image: time={duration_sec_t2i:.2f}s, in={prompt_token_count_t2i}, out={output_token_count_t2i}"
)
plt.suptitle(suptitle_text, fontsize=14, y=0.95)

# Top row - Ground Truth image and reconstruction
description_title = f'Ground Truth Description:\n"{format_prompt(query_image_description)}"'

plt.subplot(2, 2, 1)
plt.imshow(query_image)
plt.title(description_title, fontsize=12, pad=10)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(reconstructed_image_gt)
plt.title(description_title, fontsize=12, pad=10)
plt.axis('off')

# Bottom row - Text-to-Image results
plt.subplot(2, 2, 3)
plt.imshow(query_image)
plt.title("Original Image", fontsize=12, pad=10)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(reconstructed_image_gemini)
plt.title("Gemini's Generated Image", fontsize=12, pad=10)
plt.axis('off')

plt.show()