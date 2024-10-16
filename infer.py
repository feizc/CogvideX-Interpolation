import torch 
from diffusers.utils import export_to_video, load_image 
from cogvideox_interpolation.pipeline import CogVideoXInterpolationPipeline 


model_path = '/maindata/data/shared/public/multimodal/share/zhengcong.fei/ckpts/CogVideoX-5b-I2V-inter' 

pipe = CogVideoXInterpolationPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()


with open('cases/example.txt', 'r') as f:
    text_list = f.readlines()[4:]

i = 5
for text in text_list: 
    prompt = text.strip()
    print(prompt)
    first_image = load_image(image="cases/"+str(i)+'.jpg')
    last_image = load_image(image="cases/"+str(i)+str(i)+'.jpg')
    video = pipe(
        prompt=prompt,
        first_image=first_image,
        last_image=last_image,
        num_videos_per_prompt=50,
        num_inference_steps=50,
        num_frames=49,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    )[0]
    export_to_video(video[0], "cases/gen_"+str(i)+".mp4", fps=8)
    i += 1 

    