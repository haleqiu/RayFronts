"""A script to compute PCA basis for a specific encoder for a data distribution

The data can be a folder of unorganized images or any frame sequence dataset.
"""

import argparse
import os
import random
import sys

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import tqdm

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from rayfronts import datasets, image_encoders


@torch.no_grad()
def main(args = None):
  if args.seed >= 0:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    
  device = "cuda" if torch.cuda.is_available() else "cpu"

  if args.dataset == "replica":
    dataset = datasets.NiceReplicaDataset(args.dataset_path, args.dataset_scene,
                                          args.resolution)
  elif args.dataset == "rosnpy":
    dataset = datasets.RosnpyDataset(args.dataset_path, args.resolution)

  elif args.dataset == "image_folder":
    dataset = torchvision.datasets.ImageFolder(
      args.dataset_path,
      torchvision.transforms.Compose(
        [torchvision.transforms.Resize((args.resolution, args.resolution)),
        torchvision.transforms.ToTensor()]))

  dataloader = torch.utils.data.DataLoader(
    dataset, shuffle=args.dataset == "image_folder")

  encoder = image_encoders.NARadioEncoder(device=device, return_radio_features=True,
                                          model_version="radio_v2.5-l")

  features = list()
  for batch in tqdm.tqdm(dataloader):
    if np.random.rand() > args.frame_sampling:
      continue

    if args.dataset == "image_folder":
      rgb_img, _ = batch
    else:
      rgb_img = batch["rgb_img"]
    
    rgb_img = rgb_img.to(device)

    feat_img = encoder.encode_image_to_feat_map(rgb_img)
    B, FC, FH, FW = feat_img.shape
    feat_img_flat = feat_img.permute(0, 2, 3, 1).reshape(-1, FC)
    k = int(args.pixel_sampling * feat_img_flat.size(0))
    perm = torch.randperm(feat_img_flat.size(0), device=device)
    idx = perm[:k] 
    sampled_features = feat_img_flat[idx, :]
    features.append(sampled_features)

  q = args.n_comps if args.n_comps > 0 else FC
  features = torch.cat(features, dim=0)
  U,S,V = torch.pca_lowrank(features, q = q)

  torch.save(V, args.output)

  if args.vis:
    variance = S**2
    plt.plot( (torch.cumsum(variance, dim=0) / torch.sum(variance)).cpu())
    plt.title("Explained variance of PCA components")
    plt.xlabel("PCA Component idx")
    plt.ylabel("Cummulative percentage of explained variance")
    plt.show()

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Get PCA Basis")

  parser.add_argument("--dataset", "-d", type=str, required=True,
                      choices=["image_folder", "replica",
                               "rosnpy"], help="Dataset type")
  parser.add_argument("--dataset_path", "-p", type=str, required=True)
  parser.add_argument("--dataset_scene", "-s", type=str)

  parser.add_argument("--resolution", "-r", type=int, default=512,
                      help="Set the operating resolution when loading "
                           "the data. if set to None, then the original "
                           "resolution is used.")

  parser.add_argument("--output", "-o", type=str, required=True)

  parser.add_argument("--frame_sampling", "-fs", type=float, default=0.1,
                      help="Percentage of the frames to use for PCA (Roughly)")

  parser.add_argument("--pixel_sampling", "-ps", type=float, default=0.1,
                    help="Percentage of the pixels to use for PCA")

  parser.add_argument("--n_comps", type=int, default=-1,
                      help="Number of PCA components to store. "
                           "Set to -1 to compute all")

  parser.add_argument("--batch_size", "-bs", type=int, default=8)

  parser.add_argument("--vis", "-v", action="store_true")

  parser.add_argument("--seed", type=int, default=11,
                      help="Set to -1 to disable setting a seed")

  args = parser.parse_args()
  main(args = args)
