#%% Imports

import os
import io
import re
import glob
import time
import base64
import tiktoken
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multimodal_protocol import LargeMultimodalModelProtocol

#%% Helper functions

def get_encoding_stats(encoded_string):

    # Initialize tokenizer
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    # Calculate total characters
    total_chars = len(encoded_string)
    
    # Calculate total tokens
    total_tokens = len(enc.encode(encoded_string))
    
    # Parse header to get hierarchy information
    hierarchy_entries = re.findall(r'<h_p(\d+)f\d+q(?:RGB|\d+)>', encoded_string)
    patch_sizes = [int(size) for size in hierarchy_entries]

    # Count patches for each level
    patches_per_level = {}
    for size in patch_sizes:
        rgb_pattern = f'<p{size}qRGB>'
        dict_pattern = f'<p{size}q\d+>'
        count_rgb = len(re.findall(rgb_pattern, encoded_string))
        count_dict = len(re.findall(dict_pattern, encoded_string))
        patches_per_level[size] = count_rgb + count_dict
    
    num_hierarchies = len(patch_sizes)
    num_elements = 3 * sum(patches_per_level.values()) + num_hierarchies + 1

    encoding_stats = {
        'total_chars': total_chars,
        'patches_per_level': patches_per_level,
        'total_tokens': total_tokens,
        'total_<>_elements': num_elements,
    }

    return encoding_stats

def get_compression_stats(original_image, encoded_string):

    # Initialize tokenizer
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    # Calculate base64 metrics
    img_byte_arr = io.BytesIO()
    original_image.save(img_byte_arr, format=original_image.format if original_image.format else 'PNG')
    base64_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    
    base64_length = len(base64_str)
    base64_tokens = len(enc.encode(base64_str))
        
    # Calculate char encoding metrics
    encoding_length = len(encoded_string)
    encoding_tokens = len(enc.encode(encoded_string))
    
    # Calculate pixel channels
    img_array = np.array(original_image)
    pixel_channels = img_array.size  # Total number of values (height * width * channels)

    # Calculate total patches
    encoded_string_stats = get_encoding_stats(encoded_string)
    patches_per_level = encoded_string_stats['patches_per_level']
    total_patches = sum(patches_per_level.values())
    total_elements = encoded_string_stats['total_<>_elements']
    
    base64_tokens_fraction = encoding_tokens / base64_tokens
    base64_char_fraction = encoding_length / base64_length

    compression_stats = {
        'base64_length': base64_length,
        'base64_tokens': base64_tokens,
        'encoding_length': encoding_length,
        'encoding_tokens': encoding_tokens,
        'pixel_channels': pixel_channels,
        'total_patches': total_patches,
        'total_<>_elements': total_elements,
        'base64_tokens_fraction': base64_tokens_fraction,
        'base64_char_fraction': base64_char_fraction,
    }

    return compression_stats

def visualize_all_configurations(lmmp, image, encoding_configs):
    """Visualize all configurations and their progressive encoding in a single plot."""
    num_hierarchies = max(len(config['patch_sizes']) for config in encoding_configs)
    num_configs = len(encoding_configs)
    
    # Create figure with num_configs rows and max_levels + 1 columns
    fig = plt.figure(figsize=(4 * (num_hierarchies + 1), 5 * num_configs))
    fig.subplots_adjust(top=0.96)
    gs = fig.add_gridspec(num_configs, num_hierarchies + 1, hspace=0.2, wspace=0.05)
    
    # Process each configuration
    for config_idx, config in enumerate(encoding_configs):

        patch_sizes = config['patch_sizes']
        patch_fractions = config['fractions']
        patch_quantizations = config['quantizations']

        # Show original image in first column
        ax = fig.add_subplot(gs[config_idx, 0])
        ax.imshow(image)
        ax.set_title("Original")
        ax.axis('off')
        
        for hierarchy_inx in range(1, num_hierarchies + 1):
            
            curr_patch_sizes = patch_sizes[:hierarchy_inx]
            curr_patch_fractions = patch_fractions[:hierarchy_inx]
            curr_patch_quantizations = patch_quantizations[:hierarchy_inx]

            start_time = time.time()
            image_str = lmmp.image2string(image, curr_patch_sizes, curr_patch_fractions, curr_patch_quantizations)
            encoding_time = time.time() - start_time
            
            start_time = time.time()
            image_rec = lmmp.string2image(image_str)
            decoding_time = time.time() - start_time

            stats = get_compression_stats(image, image_str)

            ax = fig.add_subplot(gs[config_idx, hierarchy_inx])
            ax.imshow(image_rec)
            
            timing_str = f'encoding: {encoding_time:.2f}s decoding: {decoding_time:.2f}s'
            config_str = f'p={curr_patch_sizes}, \nf={curr_patch_fractions},\n q={curr_patch_quantizations}'
            stats_str = f'chars: {stats["encoding_length"]}, tokens: {stats["encoding_tokens"]} \nelements: {stats["total_<>_elements"]}, compression: {stats["base64_tokens_fraction"]:.2f}x'

            title_str = config_str + '\n' + timing_str + '\n' + stats_str

            ax.set_title(title_str)
            ax.axis('off')
                
    return fig


#%% Example usage

if __name__ == "__main__":

    # Paths setup
    current_file_path = os.path.abspath(__file__)
    dictionary_folder = os.path.join(os.path.dirname(current_file_path), 'patch_dicts')
    grid_image_path = os.path.join(dictionary_folder, 'patch_grid_p128_g64_8192x8192.jpg')
    
    images_folder = r'data/images'
    image_paths = glob.glob(os.path.join(images_folder, '*.jpg'))
    
    # Initialize LMMP
    lmmp = LargeMultimodalModelProtocol(grid_image_path)
    
    # Test different encoding configurations
    encoding_configs = [
        {
            'patch_sizes': [64, 32, 16],
            'fractions': [1.0, 0.8, 0.6],
            'quantizations': ['rgb', 'rgb', 'rgb'],
        },
        {
            'patch_sizes': [64, 32, 16],
            'fractions': [1.0, 0.8, 0.6],
            'quantizations': ['rgb', 'rgb', 1024],
        },
        {
            'patch_sizes': [64, 32, 16],
            'fractions': [1.0, 0.8, 0.6],
            'quantizations': [64, 64, 64],
        },
        {
            'patch_sizes': [64, 32, 16],
            'fractions': [1.0, 0.8, 0.6],
            'quantizations': [256, 256, 256],
        },
        {
            'patch_sizes': [64, 32, 16],
            'fractions': [1.0, 0.8, 0.6],
            'quantizations': [1024, 1024, 1024],
        },
        {
            'patch_sizes': [64, 32, 16],
            'fractions': [1.0, 0.8, 0.6],
            'quantizations': [64, 64, 1024],
        },
    ]
    
    # load sample image
    selected_index = np.random.randint(len(image_paths))
    image_path = image_paths[selected_index]
    image = Image.open(image_path)
    
    # Create visualization and save to file
    fig = visualize_all_configurations(lmmp, image, encoding_configs)
    fig.suptitle(f'Image Encoding with LMMP protocol\n{os.path.basename(image_path)}', fontsize=14, y=1.02)

    figure_name = os.path.basename(image_path).split('.')[0]
    figure_output_path = os.path.join('figures', f'LMMP_encoding_params_{figure_name}.png')
    fig.savefig(figure_output_path, bbox_inches='tight')

    # Print detailed statistics
    print("--------------------------------------------------------")
    print("encoding statistics:")
    print("--------------------")
    for config in encoding_configs:

        patch_sizes = config['patch_sizes']
        patch_fractions = config['fractions']
        patch_quantizations = config['quantizations']
        
        config_name = f'p={patch_sizes}, f={patch_fractions}, q={patch_quantizations}'
        print(f"{config_name}")
        print("-" * len(config_name))
        
        start_time = time.time()
        encoded_str = lmmp.image2string(image, patch_sizes, patch_fractions, patch_quantizations)
        encoding_time = time.time() - start_time
        stats = get_compression_stats(image, encoded_str)
        
        print(f"Encoding time: {encoding_time:.2f}s")
        print(f"Base64 length: {stats['base64_length']}")
        print(f"Base64 tokens: {stats['base64_tokens']}")
        print(f"string length: {stats['encoding_length']}")
        print(f"string tokens: {stats['encoding_tokens']}")
        print(f"Pixel channels: {stats['pixel_channels']}")
        print(f"Total patches: {stats['total_patches']}")
        print(f"Total <> elements: {stats['total_<>_elements']}")
        print(f"Compression ratio (vs base64):")
        print(f"  - chars : {stats['base64_char_fraction']:.2f}x")
        print(f"  - tokens: {stats['base64_tokens_fraction']:.2f}x")
        print("--------------------------------------------------------")



# %%
