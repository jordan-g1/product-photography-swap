import torch
from PIL import Image
import math

# It's assumed that 'tensor2pil' and 'pil2tensor' are available in ComfyUI's environment.
# When this code runs inside ComfyUI, it will use ComfyUI's own versions of these.
# The mock versions below are only for standalone testing or understanding.
try:
    from comfy.utils import pil2tensor, tensor2pil
except ImportError:
    print("WARNING: comfy.utils not found. Using mock pil2tensor and tensor2pil for standalone execution.")
    # Fallback mock implementations (simplified)
    import numpy as np
    def tensor2pil(tensor_image: torch.Tensor) -> Image.Image:
        if tensor_image.ndim == 4 and tensor_image.shape[0] == 1: # Handle [1,H,W,C]
            tensor_image = tensor_image.squeeze(0)
        elif tensor_image.ndim != 3: # Expect [H,W,C]
            raise ValueError(f"Mock tensor2pil expects [H,W,C] or [1,H,W,C], got {tensor_image.shape}")

        if tensor_image.is_cuda:
            tensor_image = tensor_image.cpu()
        
        np_image = tensor_image.numpy()

        if np_image.max() <= 1.0 and np_image.min() >=0.0 :
            np_image = (np_image * 255).astype(np.uint8)
        else:
            np_image = np_image.astype(np.uint8)
        
        channels = np_image.shape[-1]
        if channels == 1:
            return Image.fromarray(np_image.squeeze(-1), mode='L')
        elif channels == 3:
            return Image.fromarray(np_image, mode='RGB')
        elif channels == 4:
            return Image.fromarray(np_image, mode='RGBA')
        else:
            raise ValueError(f"Unsupported channel count for mock tensor2pil: {channels}")

    def pil2tensor(pil_image: Image.Image) -> torch.Tensor:
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        if pil_image.mode == 'L':
            np_image = np.expand_dims(np_image, axis=-1) # Add channel dim for grayscale
        return torch.from_numpy(np_image).unsqueeze(0)


class ResizeImageToNearest64AspectRatio: # Python class name
    # Define the allowed aspect ratios: Name -> (Width_Part, Height_Part)
    ALLOWED_ASPECT_RATIOS = {
        "9:16": (9, 16), "2:3": (2, 3), "3:4": (3, 4),
        "1:1": (1, 1),
        "4:3": (4, 3), "3:2": (3, 2), "16:9": (16, 9)
    }

    # Pillow resampling filter mapping
    RESAMPLE_FILTERS = {
        'lanczos': Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS,
        'nearest': Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST,
        'bilinear': Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC,
    }

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_dimension": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 64}),
                "supersample": (["true", "false"], {"default": "false"}),
                "resampling": (list(cls.RESAMPLE_FILTERS.keys()), {"default": "lanczos"}),
            },
        }

    # MODIFIED RETURN_TYPES AND RETURN_NAMES
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "width", "height", "smallest_side_dimension", "largest_side_dimension", "max_dimension_input")
    FUNCTION = "execute"
    CATEGORY = "Image/Transform" # You can change this to "WAS Suite/Image/Transform" or your preference

    def _calculate_target_dimensions(self, original_pil_image: Image.Image, max_dimension_input: int):
        original_width, original_height = original_pil_image.size
        if original_width == 0 or original_height == 0:
            return 0, 0
        
        original_ar = original_width / original_height
        viable_options = []
        
        for ar_name, (W_ratio, H_ratio) in self.ALLOWED_ASPECT_RATIOS.items():
            common_divisor = math.gcd(W_ratio, H_ratio)
            w_norm = W_ratio // common_divisor
            h_norm = H_ratio // common_divisor

            if max(w_norm, h_norm) == 0: continue 
            denominator = 64 * max(w_norm, h_norm)
            if denominator == 0: continue 

            k = math.floor(max_dimension_input / denominator)

            if k >= 1:
                calc_w = k * w_norm * 64
                calc_h = k * h_norm * 64
                if calc_w == 0 or calc_h == 0: continue
                current_ar_val = calc_w / calc_h
                viable_options.append({
                    'name': ar_name, 'width': calc_w, 'height': calc_h, 
                    'ar_value': current_ar_val, 'k': k,
                    'ar_diff': abs(current_ar_val - original_ar) 
                })

        if not viable_options:
            if "1:1" in self.ALLOWED_ASPECT_RATIOS and max_dimension_input >=64:
                 return (64, 64) 
            return original_width, original_height

        viable_options.sort(key=lambda x: (x['ar_diff'], -x['k']))
        best_option = viable_options[0]
        return best_option['width'], best_option['height']

    def _apply_pil_resize(self, image: Image.Image, target_width: int, target_height: int, 
                          supersample_str: str, resample_method: str):
        if target_width <= 0 or target_height <= 0: # Check for non-positive dimensions
            print(f"Warning: Invalid target dimensions ({target_width}x{target_height}). Returning original PIL image.")
            return image

        resample_filter = self.RESAMPLE_FILTERS.get(resample_method.lower(), self.RESAMPLE_FILTERS['lanczos'])

        if supersample_str == 'true':
            ss_width = target_width * 8
            ss_height = target_height * 8
            if ss_width > 0 and ss_height > 0:
                try:
                    image = image.resize((ss_width, ss_height), resample=resample_filter)
                except Exception as e:
                    print(f"Error during supersampling resize to ({ss_width}x{ss_height}): {e}. Skipping supersample.")
        
        try:
            resized_image = image.resize((target_width, target_height), resample=resample_filter)
        except Exception as e:
            print(f"Error during final resize to ({target_width}x{target_height}): {e}. Returning last valid image state.")
            return image
            
        return resized_image

    def execute(self, image: torch.Tensor, max_dimension: int, supersample: str, resampling: str):
        if image is None or image.nelement() == 0:
            print("Warning: Input image tensor is empty. Returning empty tensor and 0x0 dimensions.")
            # MODIFIED: Add placeholders for new outputs
            return (torch.empty_like(image) if image is not None else torch.tensor([]), 0, 0, 0, 0, max_dimension)

        pil_img_for_calc = tensor2pil(image[0]) 
        target_w, target_h = self._calculate_target_dimensions(pil_img_for_calc, max_dimension)

        if target_w == 0 or target_h == 0:
            print(f"Warning: Target dimension calculation resulted in 0. Outputting original images and 0x0 dimensions.")
            # MODIFIED: Add placeholders for new outputs
            return (image, 0, 0, 0, 0, max_dimension)

        # ADDED: Calculate new output values
        smallest_side_val = min(target_w, target_h)
        largest_side_val = max(target_w, target_h)
        max_dimension_input_val = max_dimension # This is the input value

        output_images = []
        for img_tensor_item in image: # img_tensor_item is [H,W,C]
            current_pil_img = tensor2pil(img_tensor_item)
            processed_pil_img = self._apply_pil_resize(current_pil_img, target_w, target_h, supersample, resampling)
            output_images.append(pil2tensor(processed_pil_img)) # pil2tensor returns [1,H,W,C]
        
        if not output_images:
             print("Error: No images were processed. Returning original image batch.")
             # MODIFIED: Add placeholders for new outputs, using 0 for calculated/processed dimensions
             return (image, 0, 0, 0, 0, max_dimension) 

        scaled_images_tensor = torch.cat(output_images, dim=0)
        
        # MODIFIED: Update final return statement
        return (scaled_images_tensor, target_w, target_h, smallest_side_val, largest_side_val, max_dimension_input_val)

# Node Mappings: ComfyUI will use these to register the node.
NODE_CLASS_MAPPINGS = {
    "ResizeImageToNearest64AspectRatio": ResizeImageToNearest64AspectRatio 
    # "InternalUniqueName": ActualClassName
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ResizeImageToNearest64AspectRatio": "Resize Image to Nearest 64"
    # "InternalUniqueName": "User-Friendly Display Name"
}