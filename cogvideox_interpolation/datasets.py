import json 
import torch 
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset 
import torchvision.transforms as TT  
from torchvision import transforms
from torchvision.transforms.functional import center_crop, resize 
from torchvision.transforms import InterpolationMode 
import random 
try:
    import decord
except ImportError:
    raise ImportError(
        "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
    )

decord.bridge.set_bridge("torch")

class ImageVideoDataset(Dataset): 
    def __init__(
        self,
        data_root,
        tokenizer,
        max_sequence_length: int = 226,
        height: int = 480,
        width: int = 720,
        video_reshape_mode: str = "center",
        fps: int = 8,
        stripe: int = 2,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        random_flip: Optional[float] = None,
    ) -> None:
        super().__init__() 

        with open(data_root, 'r') as f: 
            self.data_list = json.load(f)
        
        self.tokenizer = tokenizer 
        self.max_sequence_length = max_sequence_length 
        self.height = height
        self.width = width
        self.video_reshape_mode = video_reshape_mode
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end 
        self.stripe = stripe 
        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip) if random_flip else transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    
    def __len__(self):
        return len(self.data_list) 
    
    def _resize_for_rectangle_crop(self, arr):
        image_size = self.height, self.width
        reshape_mode = self.video_reshape_mode
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr
    
    def __getitem__(self, index): 
        while True: 
            try:
                video_reader = decord.VideoReader(self.data_list[index]['file_path'], width=self.width, height=self.height)
                video_num_frames = len(video_reader) 
                # print(video_num_frames, video_reader.get_avg_fps()) 
                if self.stripe * self.max_num_frames > video_num_frames: 
                    stripe = 1
                else:
                    stripe = self.stripe 

                random_range = video_num_frames - stripe * self.max_num_frames - 1
                random_range = max(1, random_range)
                start_frame = random.randint(1, random_range) if random_range > 0 else 1
                
                indices = list(range(start_frame, start_frame + stripe * self.max_num_frames, stripe)) # (end_frame - start_frame) // self.max_num_frames))
                frames = video_reader.get_batch(indices)

                # Ensure that we don't go over the limit
                frames = frames[: self.max_num_frames]
                selected_num_frames = frames.shape[0]

                # Choose first (4k + 1) frames as this is how many is required by the VAE
                remainder = (3 + (selected_num_frames % 4)) % 4
                if remainder != 0:
                    frames = frames[:-remainder]
                selected_num_frames = frames.shape[0]

                assert (selected_num_frames - 1) % 4 == 0 
                if selected_num_frames == self.max_num_frames: 
                    break 
                else:
                    index = (index + 1) % len(self.data_list) 
                    continue 
            
            except Exception as e:
                index = (index + 1) % len(self.data_list) 
                print(video_num_frames, start_frame, indices)
                print(
                    "Error encounter during audio feature extraction: ", e, 
                )
                continue

        # Training transforms
        # frames = (frames - 127.5) / 127.5
        frames = frames.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]
        frames = self._resize_for_rectangle_crop(frames) 
        frames = torch.stack([self.video_transforms(frame) for frame in frames], dim=0) 

        text_inputs = self.tokenizer(
            [self.data_list[index]['text']],
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids[0]

        return frames.contiguous(), text_input_ids

