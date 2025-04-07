import json

import torch
from spark_wan.datasets.easyvideo import EasyVideoDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
import torch.distributed as dist

def easy_collate_fn(examples):
    videos = [example["instance_video"].unsqueeze(0) for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    videos = torch.cat(videos, dim=0)
    return {
        "videos": videos,
        "prompts": prompts,
    }


def load_easyvideo_dataset(
    height: int,
    width: int,
    max_num_frames: int,
    instance_data_root: str,
    train_batch_size: int,
    dataloader_num_workers: int,
    dp_rank: int,
    dp_size: int,
) -> StatefulDataLoader:

    train_dataset = EasyVideoDataset(
        video_size=(height, width),
        num_frames=max_num_frames,
        video_data=json.load(open(instance_data_root)),
    )
    
    sampler = StatefulDistributedSampler(
        train_dataset, rank=dp_rank, num_replicas=dp_size, shuffle=True
    )
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
        collate_fn=easy_collate_fn,
        pin_memory=True,
        sampler=sampler,
    )
    return train_dataloader, sampler
