## CogvideX-Interpolation: Keyframe Interpolation with CogvideoX

CogVideoX-Interpolation is a modified pipeline based on the CogVideoX structure, designed to provide more flexibility in keyframe interpolation generation. 


<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input frame 1</td>
        <td>Input frame 2</td>
        <td>Text</td>
        <td>Generated video</td>
    </tr>
  	<tr>
	  <td>
	    <img src=cases/5.jpg width="250">
	  </td>
	  <td>
	    <img src=cases/55.jpg width="250">
	  </td>
      <td>
	    A group of people dance in the street at night, forming a circle and moving in unison, exuding a sense of community and joy. A woman in an orange jacket is actively engaged in the dance, smiling broadly. The atmosphere is vibrant and festive, with other individuals participating in the dance, contributing to the sense of community and joy. 
	  </td>
	  <td>
     		<image src=cases/gen_5.gif width="250">
	  </td>
  	</tr>
  	<tr>
	  <td>
	    <img src=cases/6.jpg width="250">
	  </td>
	  <td>
	    <img src=cases/66.jpg width="250">
	  </td>
      <td>
	    A man in a white suit stands on a stage, passionately preaching to an audience. The stage is decorated with vases with yellow flowers and a red carpet, creating a formal and engaging atmosphere. The audience is seated and attentive, listening to the speaker. 
	  </td>
	  <td>
	    <image src=cases/gen_6.gif width="250">
	  </td>
  	</tr>
  <tr>
	  <td>
	    <img src=cases/7.jpg width="250">
	  </td>
	  <td>
	    <img src=cases/77.jpg width="250">
	  </td>
      <td>
	    A man in a blue suit is laughing.
	  </td>
	  <td>
	    <img src=cases/gen_7.gif width="250">
	  </td>
  	</tr>
</table >


## Quick Start
### 1. Setup repository and environment 

Our environment is totally same with CogvideoX and you can install by: 

```
pip install -r requirement.txt
```


### 2. Download checkpoint
Download the finetuned [checkpoint](huggingface), and put it with model path variable. 

### 3. Launch the inference script!
The example input keyframe pairs are in `cases` folder. 
You can run with mini code as following or refer to `infer.py` which generate cases. 

```
from diffusers.utils import export_to_video, load_image 
from cogvideox_interpolation.pipeline import CogVideoXInterpolationPipeline

model_path = "\path\to\model\download"
pipe = CogVideoXInterpolationPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

first_image = load_image(first_image_path)
last_image = load_image(second_image_path)
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
export_to_video(video_save_path, fps=8)
```

## Light-weight finetuing
You can prepare the video-text pair data as [formation](https://github.com/feizc/CogvideX-Interpolation/blob/main/cogvideox_interpolation/datasets.py) and run the training scripts as:
```
sh train.sh 
```

## Knowledge 

The codebase is based on the awesome [CogvideoX](https://github.com/THUDM/CogVideo) and [diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py) repos.







