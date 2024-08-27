import sys
import os
import time
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class stablediffusionv1_5:
    def __init__(self, mode='gpu', stage="singlize",
                 prompt="a photo of an astronaut riding a horse on mars", 
                 img_size=(512, 512), 
                 model_path='model/bmodel/multimodality/generative/stablediffusionv1_5', 
                 tokenizer='model/pytorch/multimodality/generative/stablediffusionv1_5/tokenizer_path'):
        self.mode = mode
        self.stage = stage
        self.prompt = prompt
        self.img_size = img_size
        self.model_path = model_path
        self.tokenizer = tokenizer
        
        if mode == 'gpu':
            from diffusers import StableDiffusionPipeline
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            local_model_path = "model/pytorch/multimodality/generative/stablediffusionv1_5/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"

            self.pipeline = StableDiffusionPipeline.from_pretrained(local_model_path, torch_dtype=torch.float32).to(self.device)
        elif mode == 'tpu':
            from diffusers import PNDMScheduler
            from utils.stable_diffusion import StableDiffusionPipeline
            scheduler = PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                skip_prk_steps=True,
            )
            self.pipeline = StableDiffusionPipeline(
                scheduler = scheduler,
                model_path = self.model_path,
                stage = self.stage,
                tokenizer = self.tokenizer,
                dev_id = 0,
                controlnet_name = None,
                processor_name = None,
            ) 
        else:
            raise ValueError("Mode should be either 'gpu' or 'tpu'")

    def generate_image(self):
        height, width = self.img_size
        np.random.seed()
        image = self.pipeline(
            prompt=self.prompt,
            height=height,
            width=width,
            guidance_scale=7.5,
            num_inference_steps=50
        )
        return image
    
    def forward(self):
        if self.mode == 'gpu':
            image = self.pipeline(prompt = self.prompt).images[0]
            return image
        elif self.mode == 'tpu':
            image = self.pipeline(prompt = self.prompt,
            height = self.img_size[0],
            width = self.img_size[1],
            negative_prompt = "worst quality",
            init_image = None,
            controlnet_img = None,
            strength = 0.7,
            num_inference_steps = 50,
            guidance_scale = 7.5)
            return image
        else:
            raise ValueError("Mode should be either 'gpu' or 'tpu'")
   
    def count_flops(model, input):
        import thop
        return thop.profile(model, (input, ), verbose=False)[0] / 1e9
if __name__ == '__main__':
    mode = 'gpu' 
    model = stablediffusionv1_5(mode=mode)
    import time
    import torch
    iterations = 1 
    t_start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            # model.forward()
            pass
    elapsed_time = time.time() - t_start
    from fvcore.nn import FlopCountAnalysis
    dummy_input_unet = torch.randn(1, 4, 64, 64).to(model.device)  # Corresponding to the input shape of UNet
    dummy_input_vae = torch.randn(1, 3, 64, 64).to(model.device)   # Corresponding to the input shape of VAE
    dummy_input_text = torch.randint(0, 77, (1, 77)).to(model.device)  # Corresponding to the input of the text encoder


    dummy_timestep = torch.tensor([1.0], device=model.device)  # A hypothetical time step
    dummy_text = torch.randint(0, 77, (1, 77)).to(model.device)  # Assuming text input

    # Get encode_hidden_states (from text encoder)
    encoder_hidden_states = model.pipeline.text_encoder(dummy_text)[0]

    # The inputs required for the forward propagation of UNet include: image, time step, and encoded text hidden state
    flops_unet = FlopCountAnalysis(model.pipeline.unet, (dummy_input_unet, dummy_timestep, encoder_hidden_states))
    print(f"UNet FLOPs: {flops_unet.total()}")

    # Calculate FLOPs for VAE
    flops_vae = FlopCountAnalysis(model.pipeline.vae, dummy_input_vae)
    print(f"VAE FLOPs: {flops_vae.total()}")

    # FLOPs for calculating text encoders
    flops_text_encoder = FlopCountAnalysis(model.pipeline.text_encoder, dummy_input_text)
    print(f"Text Encoder FLOPs: {flops_text_encoder.total()}")

    # Calculate total FLOPs
    total_flops = flops_unet.total() + flops_vae.total() + flops_text_encoder.total()
    print(f"Total FLOPs: {total_flops / 1e9 * 2}")

    # Calculate the parameter count of UNet
    from thop import profile
    unet_params = sum(p.numel() for p in model.pipeline.unet.parameters())
    # flops, _ = profile(model.pipeline.unet, (model.prompt, ), verbose=False)
    print(f"UNet Parameters: {unet_params}")

    # Calculate the parameter quantity of VAE
    vae_params = sum(p.numel() for p in model.pipeline.vae.parameters())
    print(f"VAE Parameters: {vae_params}")

    # Calculate the parameter count of CLIP text encoder
    text_encoder_params = sum(p.numel() for p in model.pipeline.text_encoder.parameters())
    print(f"Text Encoder Parameters: {text_encoder_params}")

    # Calculate the total number of parameters
    total_params = unet_params + vae_params + text_encoder_params
    print(f"Total Parameters: {total_params / 1e6}")

    # flops = model.count_flops(model.pipeline, model.prompt)
    # print(f"FLOPs: {flops} G")

    # print(f"FLOPs: {flops} GFLOPs")
    # print(f"Parameters: {params} Million")
    latency = elapsed_time / iterations * 1000
    FPS = 1000 / latency
    print(f"FPS: {FPS:.2f}")
