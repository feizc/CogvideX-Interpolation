# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from cogvideox_interpolation.pipeline import CogVideoXInterpolationPipeline
from diffusers.utils import export_to_video, load_image

MODEL_CACHE = "interpolation"
MODEL_URL = "https://weights.replicate.delivery/default/feizhengcong/CogvideoX-Interpolation/model.tar"

ENV_VARS = {
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HOME": MODEL_CACHE,
    "TORCH_HOME": MODEL_CACHE,
    "HF_DATASETS_CACHE": MODEL_CACHE,
    "TRANSFORMERS_CACHE": MODEL_CACHE,
    "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
}
os.environ.update(ENV_VARS)

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        print("Loading CogVideoX weights")
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, '.')
        self.pipe = CogVideoXInterpolationPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        print("setup took: ", time.time() - start)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt", default="A man in a blue suit is laughing",
        ),
        first_image: Path = Input(
            description="Input image", default=None
        ),
        last_image: Path = Input(
            description="Input image", default=None
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=6
        ),
        num_frames: int = Input(
            description="Number of frames for the output video", default=49
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator(device="cuda").manual_seed(seed)
        first_img = load_image(str(first_image))
        last_img = load_image(str(last_image))

        video = self.pipe(
            prompt=prompt,
            first_image=first_img,
            last_image=last_img,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=generator,
        )[0]

        out_path = "/tmp/output.mp4"
        export_to_video(video[0], out_path, fps=8)
        return Path(out_path)
