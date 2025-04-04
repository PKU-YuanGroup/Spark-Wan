import json
import logging
import os
import pickle
import random
from typing import List, Optional, Tuple

import decord
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    ToTensorVideo,
)
from tqdm import tqdm


class EasyVideoDataset(Dataset):
    def __init__(
        self,
        video_data: List[Tuple[str, str]],
        video_size: Tuple[int, int] = (720, 1280),
        num_frames: int = 81,
        cache_dir: Optional[str] = "./dataset_cache",
    ):
        self.video_data = video_data
        self.video_size = video_size
        self.num_frames = num_frames

        self.transform = Compose(
            [
                ToTensorVideo(),
                CenterCropVideo(self.video_size),
                NormalizeVideo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.data = []

        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        for base_dir, json_or_pkl_path in tqdm(
            video_data, desc="Processing video data"
        ):
            file_base_name = (
                os.path.basename(json_or_pkl_path)
                + f"_{self.num_frames}_{self.video_size[0]}_{self.video_size[1]}.cache"
            )
            cache_path = os.path.join(cache_dir, file_base_name)
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as fi:
                    self.data.extend(pickle.load(fi))
                continue

            if json_or_pkl_path.endswith(".json"):
                with open(json_or_pkl_path, "r") as fi:
                    data = json.load(fi)
            else:
                with open(json_or_pkl_path, "rb") as fi:
                    data = pickle.load(fi)
            cache_data = []
            for item in tqdm(data, desc="Processing items in file", leave=False):
                video_path = os.path.join(base_dir, item["path"])
                if not os.path.exists(video_path):
                    continue

                if (
                    item["resolution"]["width"] < self.video_size[1]
                    or item["resolution"]["height"] < self.video_size[0]
                ):
                    continue
                if item["num_frames"] < self.num_frames:
                    continue

                cache_data.append(
                    {
                        "path": video_path,
                        "caption": item["cap"][0],
                        "cut": item["cut"],
                        "crop": item["crop"],
                    }
                )
            if not dist.is_initialized() or dist.get_rank() == 0:
                with open(cache_path, "wb") as fi:
                    pickle.dump(cache_data, fi)
            self.data.extend(cache_data)

        logging.info(f"Totoal data: {len(self.data)}!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            video_reader = decord.VideoReader(self.data[idx]["path"])

            start = self.data[idx]["cut"][0]
            video = video_reader.get_batch(
                list(range(start, start + self.num_frames))
            ).asnumpy()
            video = torch.from_numpy(video)  # t h w c
            video = video[
                :,
                self.data[idx]["crop"][2] : self.data[idx]["crop"][3],
                self.data[idx]["crop"][0] : self.data[idx]["crop"][1],
                :,
            ]
            video = self.transform(video)  # c t h w

        except Exception:  # If loading video failed, return a random video
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        return {"instance_video": video, "instance_prompt": self.data[idx]["caption"]}
