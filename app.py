import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
import numpy as np
from PIL import Image
import gradio as gr
import re



max_new_tokens = 64*64 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions


torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device = 'cpu' # for later use in torch.autocast
ptdtype=torch.float32
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)


def bmp12_to_rgb24(bitmap):
    r = (bitmap >> 8) & 0xF
    g = (bitmap >> 4) & 0xF
    b = bitmap & 0xF

    r = (r << 4) | r
    g = (g << 4) | g
    b = (b << 4) | b

    rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")

def rgb24_to_12(rgba_str):
    values = re.findall(r"[\d.]+", rgba_str)
    r = round(float(values[0]))
    g = round(float(values[1]))
    b = round(float(values[2]))

    pixel = (r << 16) | (g << 8) | b


    r = ((pixel >> 16) & 0xFF) >> 4
    g = ((pixel >> 8) & 0xFF) >> 4
    b = (pixel & 0xFF) >> 4
    return (r << 8) | (g << 4) | b



#load model
ckpt_path = 'ckpt10k.pt'
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)


model.eval()
model.to(device)

meta_path = 'meta.pkl'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)


first_pixel=[0xfec,0x8ee,0x160] #12 bit colors for the first pixel: sky gray, sky blue, grass green





def run(pixel_color, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #pixel = int(pixel_color[1:], 16)
    x = torch.tensor([[rgb24_to_12(pixel_color)]], dtype=torch.long, device=device)

    y = model.generate(x, max_new_tokens=4095, temperature=temperature)

    pixels = y[0, 1:].cpu().numpy()
    pixels = np.append(pixels, pixels[-1])   # duplicate last value
    img = pixels.reshape(64, 64)
    img=bmp12_to_rgb24(img)

    return img


#gradio
def generate_image(prompt, seed):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    return run(prompt, seed)


with gr.Blocks(title="Text-to-Image Model") as demo:
    gr.Markdown("# 🎨 Text-to-Image Generator")
    gr.Markdown("Based on Andrej Karpathy's NanoGPT")
    gr.Markdown("[GitHub](https://github.com/Ferrariedhgs/nanoGPT-BMP)")
    gr.Markdown("Dataset: [Nature-Landscape](https://huggingface.co/datasets/ferrariedhgs/Nature-Landscape)")
    gr.Markdown("Choose a color for the first pixel to generate an image.")

    prompt_input = gr.ColorPicker(
        label="Color",
        value="#f0e0c0"
    )
    seed_input = gr.Slider(
        label="Seed",
        minimum=0,
        maximum=2048,
        value=1337
    )
    generate_button = gr.Button("Generate Image")

    output_image = gr.Image(
        label="Generated Image",
        type="pil"
    )

    generate_button.click(
        fn=generate_image,
        inputs=[prompt_input, seed_input],
        outputs=output_image
    )

demo.launch()