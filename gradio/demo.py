# adapted from https://colab.research.google.com/drive/1jQI_7aaGLX0qCl82ez-fwziDwuWxMOS3?usp=sharing
from transformers import ViTModel
import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from transformers import ViTFeatureExtractor
import os
import gradio as gr

torch.hub.download_url_to_file('https://cdn.pixabay.com/photo/2018/08/12/16/59/ara-3601194_1280.jpg', 'parrot.jpg')
torch.hub.download_url_to_file('https://cdn.pixabay.com/photo/2016/12/13/00/13/rabbit-1903016_1280.jpg', 'rabbit.jpg')


# let's use the small DeiT model trained with a patch size of 8
# model = ViTModel.from_pretrained("nielsr/dino_deits8", add_pooling_layer=False)
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return



def visualize(image):
    image = image.convert("RGB")
    feature_extractor = ViTFeatureExtractor(do_resize=True, size=512, image_mean=[0.485, 0.456, 0.406], image_std=[0.485, 0.456, 0.406])
    img = feature_extractor(images=image, return_tensors="pt").pixel_values # we remove the batch dimension, is added later on

    outputs = model(pixel_values=img, output_attentions=True)

    attentions = outputs['attentions'][-1] # we are only interested in the attention maps of the last layer
    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)


    threshold = 0.6
    w_featmap = img.shape[-2] // model.config.patch_size
    h_featmap = img.shape[-1] // model.config.patch_size

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=model.config.patch_size, mode="nearest")[0].cpu()
    attentions = attentions.detach().numpy()

    # show and save attentions heatmaps

    plt.axis("off")
    plt.imshow(attentions[2])
    return plt

inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="plot",label="Output Image")

examples = [
    ['parrot.jpg'],
    ['rabbit.jpg']
]

title = "DINO"
description = "demo for Facebook DINO for visualizing attention maps. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training'>Emerging Properties in Self-Supervised Vision Transformers</a> | <a href='https://github.com/facebookresearch/dino'>Github Repo</a></p>"

gr.Interface(visualize, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()
