import numpy as np
from PIL import Image
import os
import pickle
import glob


def load_image_as_array(path):
    #load image
    img = Image.open(path)
    arr = np.array(img)

    #split channels
    r,g,b = np.split(arr,3,axis=2)
    r=r.reshape(-1)
    g=g.reshape(-1)
    b=b.reshape(-1)
    r = r >> 4
    g = g >> 4
    b = b >> 4


    #convert from 24bit to 12bit
    bitmap = list(map(lambda x: (x[0] << 8) | (x[1] << 4) | x[2], zip(r,g,b)))
    bitmap = np.array(bitmap).reshape([arr.shape[0], arr.shape[1]])

    return bitmap

def bmp12_to_rgb24(bitmap):
    r = (bitmap >> 8) & 0xF
    g = (bitmap >> 4) & 0xF
    b = bitmap & 0xF

    r = (r << 4) | r
    g = (g << 4) | g
    b = (b << 4) | b

    rgb = np.stack([r, g, b], axis=2).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def save_bmp(bitmap,name="img.png"):
    img = bmp12_to_rgb24(bitmap)
    path=os.path.join(os.path.dirname(os.path.realpath(__file__)),name)
    img.save(path)

def bmp_to_img(bitmap):
    img = bmp12_to_rgb24(bitmap)
    return img







def main():
    img_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),"image_0.png")
    bmp=load_image_as_array(img_path)
    save_bmp(bmp)
    
    H, W = 512, 512
    tokens_per_image = H * W

    all_tokens = []

    dir=os.path.dirname(os.path.realpath(__file__))

    img_folder="images/*.png"

    image_paths = sorted(glob.glob(img_folder))
    for p in image_paths:
        print(f'image {p}')
        img = load_image_as_array(p)        # shape (H, W)
        assert img.shape == (H, W)
        all_tokens.append(img.flatten())

    all_tokens = np.concatenate(all_tokens).astype(np.uint16)

    # split train / val by images, not tokens
    num_images = len(image_paths)
    split = int(0.9 * num_images)

    train_tokens = all_tokens[:split * tokens_per_image]
    val_tokens   = all_tokens[split * tokens_per_image:]

    train_tokens.tofile(os.path.join(dir,"data/bmp/train.bin"))
    val_tokens.tofile(os.path.join(dir,"data/bmp/val.bin"))

    with open(os.path.join(dir,"data/bmp/meta.pkl"), "wb") as f:
        pickle.dump({
            "vocab_size": int(all_tokens.max()) + 1,
            "height": H,
            "width": W,
            "tokens_per_image": tokens_per_image,
        }, f)



if __name__=="__main__":
    main()