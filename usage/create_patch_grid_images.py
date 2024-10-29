#%% Imports
 
import os
import re
import glob
import umap
import random
import numpy as np
from math import ceil
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment

#%% Create patch grid images from a bank of face images

image_bank_folder = r"data/images"
output_folder = r"patch_grid_images"
grid_sizes_to_use = [16, 32, 64, 128, 256]
patch_sizes_to_use = [8, 16, 32, 64, 128]

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of all jpg images in the bank folder
image_files = glob.glob(os.path.join(image_bank_folder, "*.jpg"))
num_source_images = len(image_files)

print(f"Found {num_source_images} jpg images in {image_bank_folder}")
print('---------------------------------')

if num_source_images == 0:
    raise ValueError(f"No jpg images found in {image_bank_folder}")

# Process each combination of grid size and patch size
for g in grid_sizes_to_use:
    for p in patch_sizes_to_use:
        # Calculate required number of patches (q = g^2)
        q = g * g
        
        # skip if output image size g*p is larger than 8192
        if g * p > 8192:
            print(f"Skipping grid size {g} and patch size {p} because the output image size would be too large")
            continue

        # Calculate how many patches to take from each image
        if q <= num_source_images:
            # If we need fewer patches than available images,
            # randomly select subset of images and take one patch from each
            selected_images = random.sample(image_files, q)
            patches_per_image = 1
        else:
            # If we need more patches than available images,
            # use all images and calculate patches needed from each
            selected_images = image_files
            patches_per_image = ceil(q / num_source_images)
        
        print(f"Creating patch grid with {g}x{g} patches of size {p}x{p}")
        print(f"total patches needed: {q}")
        print(f"resulting image size: {g*p}x{g*p}")
        print(f"Taking {patches_per_image} patches from {len(selected_images)} images")

        # Create empty grid image
        grid_image = Image.new('RGB', (p * g, p * g))
        
        # Collect patches
        collected_patches = []
        for img_path in selected_images:
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB mode if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    # Get image dimensions
                    width, height = img.size
                    
                    # Skip if image is too small for patch size
                    if width < p or height < p:
                        continue
                    
                    # Calculate valid patch positions
                    max_x = width - p
                    max_y = height - p
                    
                    # Extract random patches from this image
                    for _ in range(patches_per_image):
                        if max_x >= 0 and max_y >= 0:
                            x = random.randint(0, max_x)
                            y = random.randint(0, max_y)
                            patch = img.crop((x, y, x + p, y + p))
                            collected_patches.append(patch)
                        
                        # Break if we have enough patches
                        if len(collected_patches) >= q:
                            break
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
            
            # Break if we have enough patches
            if len(collected_patches) >= q:
                break
        
        # Ensure we have exactly q patches
        if len(collected_patches) > q:
            collected_patches = collected_patches[:q]
        
        # Place patches in grid
        for idx, patch in enumerate(collected_patches):
            row = idx // g
            col = idx % g
            grid_image.paste(patch, (col * p, row * p))
        
        # Save grid image
        output_filename = f'patch_grid_p{p}_q{q}.jpg'
        output_path = os.path.join(output_folder, output_filename)
        grid_image.save(output_path, quality=90)
        print(f"Created '{output_filename}'")
        print('---------------------------------')


#%% re order patches in grid images using t-SNE or UMAP

def extract_grid_params(filename):
    """Extract patch size (p) and total patches (q) from filename."""
    match = re.match(r'patch_grid_p(\d+)_q(\d+)\.jpg', filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    p = int(match.group(1))
    q = int(match.group(2))
    g = int(np.sqrt(q))
    return p, q, g

def extract_patches(grid_image, p, g):
    """Extract patches from grid image into numpy array."""
    patches = []
    for i in range(g):
        for j in range(g):
            patch = np.array(grid_image.crop((j*p, i*p, (j+1)*p, (i+1)*p)))
            patches.append(patch)
    return np.array(patches)

def compute_embedding(patches, method='tsne'):
    """Compute 2D embedding of patches."""
    # Flatten patches to 2D array (q × (p*p*3))
    flat_patches = patches.reshape(len(patches), -1)
    
    # Compute embedding
    if method == 'tsne':
        embedding = TSNE(n_components=2, random_state=42).fit_transform(flat_patches)
    elif method == 'umap':
        embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(flat_patches)
    else:
        raise ValueError(f"Unknown embedding method: {method}")
    
    # Scale to [0,1] range
    scaler = MinMaxScaler()
    return scaler.fit_transform(embedding)

def assign_grid_positions(embedding, g):
    """Assign patches to grid positions using Hungarian algorithm."""
    # Create grid coordinates
    x = np.linspace(0, 1, g)
    y = np.linspace(0, 1, g)
    xx, yy = np.meshgrid(x, y)
    grid_coords = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Compute cost matrix as Euclidean distances
    cost_matrix = np.zeros((len(embedding), len(grid_coords)))
    for i in range(len(embedding)):
        cost_matrix[i] = np.sum((grid_coords - embedding[i])**2, axis=1)
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    return col_ind

def reorder_patch_grid(input_path, output_folder, embedding_method='tsne'):
    """Reorder patches in grid image using specified embedding method."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load image and extract parameters
    filename = os.path.basename(input_path)
    p, q, g = extract_grid_params(filename)
    grid_image = Image.open(input_path)
    
    print(f"Processing {filename}")
    print(f"Patch size: {p}x{p}, Grid size: {g}x{g}, Total patches: {q}")
    
    # Extract patches
    patches = extract_patches(grid_image, p, g)
    print("Extracted patches")
    
    # Compute embedding
    print(f"Computing {embedding_method} embedding...")
    embedding = compute_embedding(patches, method=embedding_method)
    print("Computed embedding")
    
    # Assign new positions
    print("Assigning new positions...")
    new_positions = assign_grid_positions(embedding, g)
    print("Positions assigned")
    
    # Create new grid image
    new_grid = Image.new('RGB', (p * g, p * g))
    for idx, pos in enumerate(new_positions):
        row = pos // g
        col = pos % g
        patch = Image.fromarray(patches[idx])
        new_grid.paste(patch, (col * p, row * p))
    
    # Save new grid
    output_filename = os.path.splitext(filename)[0] + f'_reordered_{embedding_method}.jpg'
    output_path = os.path.join(output_folder, output_filename)
    new_grid.save(output_path, quality=90)
    print(f"Saved reordered grid as '{output_filename}'")
    print('---------------------------------')

# Example usage
input_folder = r"patch_grid_images"
output_folder = r"patch_grid_images"

# Process all patch grid images in the input folder
for filename in os.listdir(input_folder):
    if filename.startswith('patch_grid_') and not filename.endswith('_reordered.jpg'):
        input_path = os.path.join(input_folder, filename)
        p, q, g = extract_grid_params(filename)

        if q > 8192:
            print(f"Skipping {filename} because it has too many patches ({q} > 8192)")
            continue

        # Try both TSNE and UMAP
        for method in ['tsne', 'umap']:
            reorder_patch_grid(input_path, output_folder, embedding_method=method)

print('---------------------------------')
print("All patch grid images reordered")
print('---------------------------------')

# %%
