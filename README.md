# Image_Generator_Pytorch_HuggingFace_SDXL
A Python-based AI Image Generator running locally on Apple Silicon (M-series chips) using Stable Diffusion XL and Hugging Face Diffusers. Create images directly from text prompts in Jupyter Notebook.

🎨 Python AI Image Generator (Stable Diffusion XL on Apple Silicon)Project OverviewThis project provides a simple yet powerful Python-based AI Image Generator that runs entirely locally on Apple Silicon (M-series) chips, like your Mac Mini M4. Leveraging the advanced Stable Diffusion XL (SDXL) model via Hugging Face's Diffusers library and PyTorch's Metal Performance Shaders (MPS) backend, it allows you to generate high-quality images directly from text prompts within a Jupyter Notebook environment, without needing any cloud APIs.It's perfect for anyone looking to explore generative AI, create unique visuals, or simply experiment with cutting-edge diffusion models on their macOS machine.✨ FeaturesLocal Generation: All image generation happens on your machine, ensuring privacy and no API costs.Stable Diffusion XL (SDXL) Support: Utilizes the state-of-the-art SDXL base model for high-quality image output.Apple Silicon Optimization: Fully optimized to harness the power of your M-series chip's GPU via PyTorch's MPS backend.Jupyter Notebook Interface: Interactive and easy-to-use within a Jupyter Notebook, allowing for step-by-step execution and immediate visualization of results.Unique Filenames: Generated images are saved with unique timestamped filenames, preventing overwrites.Automatic Model Download: Downloads the SDXL model automatically on the first run and caches it locally for future use.⚙️ RequirementsHardwareMac Mini (or any Mac with an M-series chip): Specifically tested on a Mac Mini M4.Unified Memory (RAM):Minimum: 8GB (for smaller models or very low-resolution output).Recommended: 16GB or more for optimal performance with Stable Diffusion XL, enabling higher resolutions and faster generation. The more unified memory you have, the better the experience.Storage: At least 100GB of free SSD space for model downloads and generated images. SDXL alone can take up ~10GB.SoftwareOperating System: macOSPython: Version 3.8 - 3.12 (tested with 3.12.7).Jupyter Notebook: Or Jupyter Lab, installed and configured.Virtual Environment (Recommended): venv or conda for dependency management.🚀 Getting StartedFollow these steps to set up and run the image generator in your Jupyter Notebook.1. Clone the Repository (or Download Code)If you have this code in a repository, start by cloning it:git clone https://github.com/YourUsername/your-repo-name.git
cd your-repo-name
Alternatively, just ensure all the provided Python code cells are saved into a .ipynb file (e.g., ai_image_generator.ipynb).2. Create a Virtual Environment (Recommended)It's best practice to create a virtual environment to manage project dependencies.python3 -m venv venv_image_gen
source venv_image_gen/bin/activate
(If you use Anaconda/Miniconda: conda create -n image_gen_env python=3.12 then conda activate image_gen_env)3. Install DependenciesOpen your Jupyter Notebook and paste the content of cell 1 into the first cell. Run it.# cell 1: Install Necessary Libraries
!pip install torch torchvision torchaudio
!pip install diffusers transformers accelerate
!pip install huggingface_hub[hf_xet] # For potentially faster large file downloads
!pip install tqdm # For download progress bars
!pip install Pillow ipywidgets # Pillow for image saving, ipywidgets for Jupyter widgets

print("Cell 1: Installation complete. Restart kernel if prompted by Jupyter after major installs.")
Important: If Jupyter prompts you to restart the kernel after installing major libraries, please do so.4. Run the Jupyter Notebook CellsCopy the remaining cells (Cell 2, Cell 3, Cell 4, Cell 5) into subsequent cells in your Jupyter Notebook and run them sequentially.cell 2: Import Libraries(Copy and run this cell)# cell 2: Import Libraries
import torch
from diffusers import DiffusionPipeline
import os
from PIL import Image
from IPython.display import display # For displaying images directly in Jupyter Notebook
import datetime # Import the datetime module for timestamps

print("Cell 2: Imports complete.")
cell 3: Device Configuration(Copy and run this cell)# cell 3: Device Configuration
if torch.backends.mps.is_available():
    device = "mps"
    print(f"Cell 3: Using Apple Silicon MPS backend for GPU acceleration: {device}")
else:
    device = "cpu"
    print("Cell 3: Apple Silicon MPS backend not available. Falling back to CPU. "
          "Image generation will be significantly slower.")
    print("Please ensure your PyTorch installation supports MPS for optimal performance.")
cell 4: Load the Stable Diffusion Model(Copy and run this cell)# cell 4: Load the Stable Diffusion Model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

print(f"Cell 4: Loading Stable Diffusion model: {model_id}. This may take a while for the first download (~5-10GB).")
print("If you don't see a progress bar, the download is still happening in the background.")
print("You might see progress updates in the terminal where you launched Jupyter.")

try:
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
    pipe = pipe.to(device)

    # Optional: If you want to use a refiner model for better quality (requires more VRAM/unified memory)
    # from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
    # refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16)
    # refiner = refiner.to(device)
    # pipe.refiner = refiner # Attach refiner to the base pipeline if you plan to use it

    print("Cell 4: Stable Diffusion model loaded successfully and moved to MPS.")
    print("Unified Memory (RAM) Recommendation for SDXL: 16GB or more for best performance and stability.")

except Exception as e:
    print(f"Cell 4: Error loading the model: {e}")
    print("Please check your internet connection for the initial download, "
          "and ensure you have enough disk space and unified memory.")
    pipe = None # Ensure pipe is None if loading fails
First Run Alert: The first time you run Cell 4, the large SDXL model will be downloaded. This can take a significant amount of time (5-10 GB) depending on your internet connection. Be patient! You might see "Error displaying widget" messages in Jupyter; this just means the visual progress bar isn't rendering, but the download is still active. Check your terminal where Jupyter was launched for text-based progress updates.cell 5: Image Generation Loop(Copy and run this cell)# cell 5: Image Generation Loop
if pipe is None:
    print("Model was not loaded in Cell 4. Cannot proceed with image generation.")
else:
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCell 5: Starting Image Generation Session.")
    print(f"Images will be saved to '{output_dir}'.")
    print("Enter your text prompt to generate an image. Type 'exit' or 'quit' to end.")

    while True:
        prompt = input("Enter prompt (or 'exit'/'quit'): ").strip()

        if prompt.lower() in ['exit', 'quit']:
            print("Exiting image generation session. Goodbye!")
            break

        if not prompt:
            print("Prompt cannot be empty. Please try again.")
            continue

        print(f"Generating image for prompt: '{prompt}'...")
        try:
            generated_image = pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_prompt_part = "".join(c for c in prompt if c.isalnum() or c in (' ', '.', '_')).strip()
            if len(clean_prompt_part) > 50:
                clean_prompt_part = clean_prompt_part[:50].strip() + "..."
            
            file_name = os.path.join(output_dir, f"image_{timestamp}_{clean_prompt_part}.png")
            file_name = file_name.replace(" ", "_").replace("...", "")
            
            generated_image.save(file_name)
            print(f"Image saved to: {file_name}")
            display(generated_image)

        except Exception as e:
            print(f"An error occurred during image generation: {e}")
            print("You might be running out of unified memory, or there's an issue with the model.")
💡 Tips for Better Generations & AccuracyPrompt Engineering:Be Specific & Concise: Prioritize key subjects, actions, and styles at the beginning of your prompt.Use Strong Descriptors: Employ evocative adjectives (e.g., hyperdetailed, cinematic, ethereal).Styles & Artists: Add terms like photorealistic, concept art, oil painting, or artist names (e.g., by Greg Rutkowski).Negative Prompts: While not directly entered in this interface, the model implicitly benefits from training on various negative concepts. For finer control, you can explore the diffusers documentation for how to pass negative prompts directly if you were to modify the pipeline call.Parameters:num_inference_steps: (Currently 30) Increasing this to 50 or 100 can improve detail and fidelity but will increase generation time.guidance_scale: (Currently 7.5) Experiment with values between 7 and 12. Higher values make the image adhere more strictly to the prompt, lower values allow more creativity.SDXL Refiner (Advanced):The SDXL ecosystem includes a refiner model for an additional pass to enhance details and realism. If you have 16GB or more unified memory, uncomment and utilize the refiner in cell 4 for significantly better output quality.Long Prompts: Remember that the CLIP text encoder has a 77-token limit. If your prompt is longer, the latter part will be truncated. Focus your most critical details at the beginning.⚠️ Troubleshooting"Error displaying widget": This is a common Jupyter issue for interactive progress bars. The download/generation is still happening in the background. Check your terminal where Jupyter was launched for potential console output.OutOfMemoryError: This means your Mac's unified memory (RAM) is insufficient for the model or resolution you're trying to generate.Solution: Close other applications, try a smaller num_inference_steps, or consider using a less memory-intensive Stable Diffusion model (e.g., stabilityai/stable-diffusion-2-1-base instead of SDXL).Very Slow Generation:Ensure cell 3 correctly reports device = "mps". If it's cpu, your PyTorch isn't leveraging the GPU. Re-check PyTorch installation for MPS compatibility.The first run of Cell 4 will always be slow due to model download. Subsequent runs will be faster.Model Download Stuck/Failed: Check your internet connection. Ensure you have enough disk space.📄 LicenseThis project is open-source and available under the MIT License.🙏 AcknowledgementsStability AI for developing the amazing Stable Diffusion models.Hugging Face for the Diffusers and Transformers libraries, which make AI development accessible.PyTorch for the robust deep learning framework and Apple Silicon (MPS) support.
