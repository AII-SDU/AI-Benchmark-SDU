import argparse
import os
import numpy as np
from stable_diffusion import StableDiffusionPipeline
from diffusers import PNDMScheduler

provided_img_size = [(128, 384), (128, 448), (128, 512), (192, 384), (192, 448), (192, 512), (256, 384), 
            (256, 448), (256, 512), (320, 384), (320, 448), (320, 512), (384, 384), (384, 448), 
            (384, 512), (448, 448), (448, 512), (512, 512), (512, 576), (512, 640), (512, 704),
            (512, 768), (512, 832), (512, 896), (768, 768), (384, 128), (448, 128), (512, 128),
            (384, 192), (448, 192), (512, 192), (384, 256), (448, 256), (512, 256), (384, 320),
            (448, 320), (512, 320), (448, 384), (512, 384), (512, 448), (576, 512), (640, 512), 
            (704, 512), (768, 512), (832, 512), (896, 512)]

def run(engine, args):
    if args.img_size:
        height, width = args.img_size
        if (height, width) not in provided_img_size:
            print(f'{height},{width} is not supported.')
    else:
        print('Please provide image size using --img_size.')
    if args.prompt:
        np.random.seed()
        image = engine(
            prompt = args.prompt,
            height = height,
            width = width,
            negative_prompt = args.neg_prompt,
            init_image = args.init_img,
            controlnet_img = args.controlnet_img,
            strength = args.strength,
            num_inference_steps = args.num_inference_steps,
            guidance_scale = args.guidance_scale
        )
    return image

def load_pipeline(args):
    scheduler = PNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        skip_prk_steps = True,
    )
    pipeline = StableDiffusionPipeline(
        scheduler = scheduler,
        model_path = args.model_path,
        stage = args.stage,
        controlnet_name = args.controlnet_name,
        processor_name = args.processor_name,
        dev_id = args.dev_id,
        tokenizer = args.tokenizer
    )
    return pipeline

def parse_img_size(img_size_str):
    try:
        height, width = map(int, img_size_str.split(','))
        return height, width
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid image size format. Please use 'height,width'.")

