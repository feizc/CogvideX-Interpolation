
def cogvideox_infer(): 
    import torch
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image 
    model_path = '/maindata/data/shared/multimodal/michael.fan/ckpts/THUDM/CogVideoX-5b-I2V-turbo-inter' 
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    with open('cases/example.txt', 'r') as f:
        text_list = f.readlines()
    # print(text_list) 
    i = 1 
    for text in text_list: 
        prompt = text.strip()
        print(prompt)
        image = load_image(image="cases/"+str(i)+'.jpg')

        video = pipe(
            prompt=prompt,
            image=image,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

        export_to_video(video, "cases/gen_"+str(i)+".mp4", fps=8)
        i += 1 


def build_dataset():
    import json 
    from transformers import AutoTokenizer

    data_path = "video600w.json"
    model_path = '/maindata/data/shared/public/multimodal/share/zhengcong.fei/ckpts/CogVideoX-5b-I2V'
    with open(data_path, 'r') as f: 
        data_list = json.load(f)
    print(data_list[10:20]) 

    from cogvideox_interpolation.datasets import ImageVideoDataset 
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, subfolder="tokenizer", 
    )
    dataset = ImageVideoDataset(data_root=data_path, tokenizer=tokenizer) 
    print(dataset[0][0].size(), dataset[0][1].size())
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )
    for item in train_dataloader: 
        print(item[0].size(), item[1].size())
        break 


def extract_image(): 
    import cv2 
    for i in range(1, 5): 
        video_path = 'cases/' + str(i) + '.mp4'
        save_path = 'cases/' + str(i) + '.jpg'
        save_path_2 = 'cases/' + str(i) + str(i) + '.jpg'
        cap = cv2.VideoCapture(video_path) 
        ret, frame = cap.read() 
        cv2.imwrite(save_path, frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 49) 
        ret, frame = cap.read() 
        cv2.imwrite(save_path_2, frame)


def test_pipeline(): 
    import torch 
    from diffusers.utils import export_to_video, load_image 
    from cogvideox_interpolation.pipeline import CogVideoXInterpolationPipeline 
    model_path = '/maindata/data/shared/public/multimodal/share/zhengcong.fei/ckpts/CogVideoX-5b-I2V' 
    pipe = CogVideoXInterpolationPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()


    with open('cases/example.txt', 'r') as f:
        text_list = f.readlines()
    # print(text_list) 
    i = 1 
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



def action_dataset_build():
    import pandas as pd 
    import json 
    csv_path = '/maindata/data/shared/public/aigame/xujing/all_preprocess_results/training_data_600w_filterred_clean_captioned_summary_with_action_balanced.csv'
    chunksize = 10000
    data_list = []
    count = 0 

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        for index, row in chunk.iterrows(): 
            # print(row)
            if row['fps'] > 31.0:
                continue 
            if row['num_frames'] < 100:
                continue 
            if row['num_frames'] > 200: 
                continue 
            data_list.append(
                {
                    "file_path": row['path'],
                    "text": row['text'] 
                }
            )
            #print(row['num_frames'])
            # read_image_from_tar(row['path'])
            #print(row['text'])
            #break 
        print(len(data_list))
        # break 
    print(len(data_list))
    with open("video600w.json", 'w') as f:
        json.dump(data_list, f) 

# action_dataset_build() 

# extract_image()
build_dataset()
# cogvideox_infer()
# test_pipeline()


