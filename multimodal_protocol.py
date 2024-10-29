#%% Imports

import os
import re
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#%% Large Multimodal Model Protocol (LMMP) main class

class LargeMultimodalModelProtocol:
    """Large Multimodal Model Protocol - Image encoding/decoding class"""
    
    def __init__(self, grid_image_path='patch_grid_p128_g64_8192x8192.jpg'):
        self.grid_image_path = grid_image_path
        self._init_patch_dictionary()
    
    def _init_patch_dictionary(self):
        """Initialize patch dictionary if grid_image_path is provided"""
        if not self.grid_image_path:
            self.patch_dict = None
            return
            
        # Parse grid parameters from filename
        filename = os.path.basename(self.grid_image_path)
        params = re.match(r'patch_grid_p(\d+)_g(\d+)_(\d+)x(\d+)_.*\.jpg', filename)
        if not params:
            raise ValueError("Invalid grid filename format")
            
        self.master_patch_size = int(params.group(1))
        self.grid_size = int(params.group(2))
        self.total_patches = self.grid_size * self.grid_size
        
        # Load and verify the master grid image
        self.master_grid = np.array(Image.open(self.grid_image_path))
        expected_dims = (self.grid_size * self.master_patch_size,
                        self.grid_size * self.master_patch_size)
        if self.master_grid.shape[:2] != expected_dims:
            raise ValueError(f"Grid image dimensions mismatch. Expected {expected_dims}")
        
        # Initialize cache for different patch size/quantization combinations
        self.patch_cache = {}
        
    def _get_dictionary(self, patch_size, quantization):
        """Get or create patch dictionary for given size and quantization"""
        cache_key = (patch_size, quantization)
        if cache_key in self.patch_cache:
            return self.patch_cache[cache_key]
        
        grid_dim = int(np.sqrt(quantization))
        stride = self.grid_size // grid_dim
        
        patches = {}
        patch_array = np.zeros((grid_dim, grid_dim, patch_size, patch_size, 3), dtype=np.uint8)
        
        for i in range(grid_dim):
            for j in range(grid_dim):
                start_row = i * stride * self.master_patch_size
                start_col = j * stride * self.master_patch_size
                patch = self.master_grid[
                    start_row:start_row + patch_size,
                    start_col:start_col + patch_size
                ]
                patches[(i, j)] = patch
                patch_array[i, j] = patch
        
        self.patch_cache[cache_key] = (patches, patch_array)
        return patches, patch_array
    
    def _find_best_match(self, target_patch, patch_array):
        """Find best matching patch using vectorized operations"""
        target = target_patch.reshape(1, 1, *target_patch.shape)
        mse = np.mean((patch_array.astype(np.float32) - target) ** 2, axis=(2, 3, 4))
        h_idx, w_idx = np.unravel_index(np.argmin(mse), mse.shape)
        return h_idx, w_idx
    
    def calculate_patch_mae(self, original_image, reconstructed_image, patch_size):
        """Calculate Mean Absolute Error for each patch."""
        # Convert PIL images to numpy arrays if needed
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        if isinstance(reconstructed_image, Image.Image):
            reconstructed_image = np.array(reconstructed_image)
        
        height, width, _ = original_image.shape
        mae_values = np.zeros((height // patch_size, width // patch_size))
        
        for row in range(0, height, patch_size):
            for col in range(0, width, patch_size):
                original_patch = original_image[row:row+patch_size, col:col+patch_size]
                reconstructed_patch = reconstructed_image[row:row+patch_size, col:col+patch_size]
                if original_patch.shape == reconstructed_patch.shape and original_patch.shape[:2] == (patch_size, patch_size):
                    mae = np.mean(np.abs(original_patch - reconstructed_patch))
                    mae_values[row // patch_size, col // patch_size] = mae
        
        return mae_values
    
    def encode_hierarchy_rgb(self, original_image, current_reconstruction, patch_size, fraction):
        """Encode a single hierarchy level using RGB averaging"""

        height, width = original_image.shape[:2]
        encoded_str = ""
        level_reconstruction = current_reconstruction.copy()
        
        def rgb_to_hex(rgb):
            return '{:02X}{:02X}{:02X}'.format(rgb[0], rgb[1], rgb[2])
        
        # Calculate patch-wise mean absolute error if needed
        if fraction < 1.0:
            # Select patches with highest error
            mae_values = self.calculate_patch_mae(original_image, current_reconstruction, patch_size)            
            num_patches = int(np.ceil(mae_values.size * fraction))
            selected = np.argpartition(mae_values.ravel(), -num_patches)[-num_patches:]
        else:
            selected = None
        
        # Process patches - using original image for patch values
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                if fraction < 1.0:
                    patch_idx = (i//patch_size) * (width//patch_size) + (j//patch_size)
                    if patch_idx not in selected:
                        continue
                
                patch = original_image[i:i+patch_size, j:j+patch_size]
                if patch.shape[:2] == (patch_size, patch_size):
                    avg_color = np.mean(patch, axis=(0, 1)).astype(np.uint8)
                    hex_color = rgb_to_hex(avg_color)
                    encoded_str += f"<p{patch_size}qRGB> <h{i}w{j}> <0x{hex_color}>\n"
                    level_reconstruction[i:i+patch_size, j:j+patch_size] = avg_color
        
        return encoded_str, level_reconstruction
    
    def encode_hierarchy_pq(self, original_image, current_reconstruction, patch_size, fraction, quantization):
        """Encode a single hierarchy level using patch dictionary"""

        height, width = original_image.shape[:2]
        encoded_str = ""
        level_reconstruction = current_reconstruction.copy()
        
        # Get or create patch dictionary for this size/quantization
        patches, patch_array = self._get_dictionary(patch_size, quantization)
        
        # Calculate patch-wise mean absolute error if needed
        if fraction < 1.0:
            # Select patches with highest error
            mae_values = self.calculate_patch_mae(original_image, current_reconstruction, patch_size)            
            num_patches = int(np.ceil(mae_values.size * fraction))
            selected = np.argpartition(mae_values.ravel(), -num_patches)[-num_patches:]
        else:
            selected = None
        
        # Process patches - matching against original image patches
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                if fraction < 1.0:
                    patch_idx = (i//patch_size) * (width//patch_size) + (j//patch_size)
                    if patch_idx not in selected:
                        continue
                
                patch = original_image[i:i+patch_size, j:j+patch_size]
                if patch.shape[:2] == (patch_size, patch_size):
                    h_idx, w_idx = self._find_best_match(patch, patch_array)
                    hex_indices = f"{h_idx:02X}{w_idx:02X}"
                    encoded_str += f"<p{patch_size}q{quantization}> <h{i}w{j}> <0x{hex_indices}>\n"
                    level_reconstruction[i:i+patch_size, j:j+patch_size] = patches[(h_idx, w_idx)]
        
        return encoded_str, level_reconstruction

    def image2string(self, image, patch_sizes, patch_fractions, patch_quantizations):
        """
        Convert image to string representation using mixed encoding schemes
        
        Args:
            image: PIL Image or numpy array
            patch_sizes: List of patch sizes for each hierarchy level
            patch_fractions: List of fractions for each hierarchy level
            patch_quantizations: List of quantization values ('rgb' or int) for each level
            
        Returns:
            str: Encoded image string
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        height, width = image.shape[:2]
        encoded_str = "<image>\n"
        encoded_str += f"<s{height}x{width}>\n"
        
        # Add hierarchy information
        for size, fraction, quant in zip(patch_sizes, patch_fractions, patch_quantizations):
            if quant == 'rgb':
                encoded_str += f"<h_p{size}f{int(fraction * 100)}qRGB> "
            else:
                encoded_str += f"<h_p{size}f{int(fraction * 100)}q{quant}> "
        encoded_str = encoded_str.rstrip() + "\n"
        
        # Process each hierarchy level
        reconstruction = np.zeros_like(image)
        
        for size, fraction, quant in zip(patch_sizes, patch_fractions, patch_quantizations):
            if quant == 'rgb':
                level_str, reconstruction = self.encode_hierarchy_rgb(image, reconstruction, size, fraction)
            else:
                level_str, reconstruction = self.encode_hierarchy_pq(image, reconstruction, size, fraction, quant)
            
            encoded_str += level_str
        
        encoded_str += "</image>"
        return encoded_str

    def string2image(self, encoded_string):
        """
        Convert encoded string back to image, processing patches in sequence
        
        Args:
            encoded_string: Encoded image string
            
        Returns:
            PIL.Image: Reconstructed image
        """

        # remove everything before the first <image> tag
        encoded_string = encoded_string[encoded_string.find("<image>"):]

        if not encoded_string.startswith("<image>"):
            raise ValueError("Invalid encoded string: missing <image> tag")
        
        # Parse dimensions
        dims_match = re.search(r'<s(\d+)x(\d+)>', encoded_string)
        if not dims_match:
            raise ValueError("Invalid header: missing dimensions")
        height, width = map(int, dims_match.groups())
        
        # Initialize output image
        output_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Process all patches in sequence
        # Match both RGB and dictionary patches
        patch_pattern = '|'.join([
            r'<p(\d+)qRGB> <h(\d+)w(\d+)> <0x([0-9A-Fa-f]{6})>',  # RGB format
            r'<p(\d+)q(\d+)> <h(\d+)w(\d+)> <0x([0-9A-Fa-f]{4})>'  # Dictionary format
        ])
        
        all_patches = re.finditer(patch_pattern, encoded_string)
        
        for match in all_patches:
            if match.group(1) is not None:
                # RGB patch
                size, row, col, hex_color = match.groups()[:4]
                size, row, col = map(int, (size, row, col))
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                output_image[row:row+size, col:col+size] = [r, g, b]
            else:
                # Dictionary patch
                size, quant, row, col, hex_indices = match.groups()[4:]
                size = int(size)
                quant = int(quant)
                row = int(row)
                col = int(col)
                h_idx = int(hex_indices[0:2], 16)
                w_idx = int(hex_indices[2:4], 16)
                
                patches, _ = self._get_dictionary(size, quant)
                patch = patches[(h_idx, w_idx)]
                output_image[row:row+size, col:col+size] = patch
        
        return Image.fromarray(output_image)
    
#%% Example usage

if __name__ == "__main__":

    current_file_path = os.path.abspath(__file__)
    dictionary_folder = os.path.join(os.path.dirname(current_file_path), 'patch_dicts')
    grid_image_path = os.path.join(dictionary_folder, 'patch_grid_p128_g64_8192x8192.jpg')
    
    images_folder = r'data/images'
    image_paths = glob.glob(os.path.join(images_folder, '*.jpg'))
    image_filenames = [os.path.basename(image_path) for image_path in image_paths]
    print('---------------------------------------')
    for filename in image_filenames:
        print(filename)
    print('---------------------------------------')

    selected_filename = np.random.choice(image_filenames)
    image_path = os.path.join(images_folder, selected_filename)

    # initialize LMMP object
    LMMP = LargeMultimodalModelProtocol(grid_image_path)

    # set the configuration
    patch_sizes = [128, 32, 16]
    patch_fractions = [1.0, 0.8, 0.5]
    patch_quantizations = ['rgb', 'rgb', 256]

    # Load original image
    image_PIL = Image.open(image_path)

    # Encode image
    encoded_str = LMMP.image2string(image_PIL, patch_sizes, patch_fractions, patch_quantizations)

    # Decode image
    image_PIL_rec = LMMP.string2image(encoded_str)

    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(image_PIL)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(image_PIL_rec)
    ax2.set_title("Reconstructed Image")
    ax2.axis("off")

# %%
