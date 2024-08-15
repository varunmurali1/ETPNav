#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import sys

import argparse
import numpy as np
import json
import math
import h5py
import copy
from PIL import Image
import time
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

from transformers import Dinov2Backbone, AutoImageProcessor
import clip

VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint

def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]      # load all scans
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids

def build_feature_extractor(model_name, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load(model_name, device='cuda')
    model.eval()
    return model, preprocess, device

def process_features(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.output_features == 'clip':
        model, img_transforms, device = build_feature_extractor(args.model_name, args.checkpoint_file)
    elif args.output_features == 'dino':
        # get last 12 layers
        model = Dinov2Backbone.from_pretrained("facebook/dinov2-base", out_indices=[-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]).cuda().eval()
        image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    num = 0
    with h5py.File(args.img_db, 'r') as f:
        for scan_id, viewpoint_id in scanvp_list:
            data = f[f'{scan_id}_{viewpoint_id}'][...].astype('uint8')
            images = []
            for i in range(VIEWPOINT_SIZE):
                image = data[i]
                image = Image.fromarray(image[:, :, ::-1]) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)

            if args.output_features == 'clip':
                images = torch.stack([img_transforms(image) for image in images], 0).to(device)
            elif args.output_features == 'dino':
                images = image_processor(images, return_tensors="pt")['pixel_values'].to(device)
            fts, logits = [], []
            for k in range(0, len(images), args.batch_size):
                if args.output_features == 'clip':
                    patch_fts, cls_fts, layer_fts = model.encode_image(images[k: k+args.batch_size])
                    layer_fts = layer_fts.permute(0, 1, 3, 2)
                    layer_fts = layer_fts[:, :, :, 1:]
                    b_fts = layer_fts # B, N, C, HW
                elif args.output_features == 'dino':
                    b_fts = []
                    layer_fts = model(images[k: k+args.batch_size]).feature_maps
                    layer_fts = [layer_ft.unsqueeze(0) for layer_ft in layer_fts]
                    layer_fts = torch.cat(layer_fts, dim=0)
                    layer_fts = layer_fts.permute(1, 0, 2, 3, 4)
                    layer_fts = layer_fts.reshape(layer_fts.size(0), layer_fts.size(1), layer_fts.size(2), -1)
                    b_fts = layer_fts  # B, N, C, HW
                b_fts = b_fts.data.cpu().numpy()
                fts.append(b_fts)
            fts = np.concatenate(fts, 0)
            out_queue.put((scan_id, viewpoint_id, fts, logits))
            num += 1
            print(num)

    out_queue.put(None)

def build_feature_file(args):
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)
    
    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.output_features == 'clip':
        model, img_transforms, device = build_feature_extractor(args.model_name, args.checkpoint_file)
    elif args.output_features == 'dino':
        # get last 12 layers
        model = Dinov2Backbone.from_pretrained("facebook/dinov2-base", out_indices=[-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]).cuda().eval()
        image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    
    progress_bar = ProgressBar(maxval=len(scanvp_list))
    progress_bar.start()
    num_finished_vps = 0
    
    with h5py.File(args.output_file, 'w') as outf:
        with h5py.File(args.img_db, 'r') as f:
            for scan_id, viewpoint_id in scanvp_list:
                data = f[f'{scan_id}_{viewpoint_id}'][...].astype('uint8')
                images = []
                for i in range(VIEWPOINT_SIZE):
                    image = data[i]
                    image = Image.fromarray(image[:, :, ::-1]) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)

                if args.output_features == 'clip':
                    images = torch.stack([img_transforms(image) for image in images], 0).to(device)
                elif args.output_features == 'dino':
                    images = image_processor(images, return_tensors="pt")['pixel_values'].to(device)
                fts, logits = [], []
                for k in range(0, len(images), args.batch_size):
                    if args.output_features == 'clip':
                        patch_fts, cls_fts, layer_fts = model.encode_image(images[k: k+args.batch_size])
                        layer_fts = layer_fts.permute(0, 1, 3, 2)
                        layer_fts = layer_fts[:, :, :, 1:]
                        b_fts = layer_fts # B, N, C, HW
                    elif args.output_features == 'dino':
                        b_fts = []
                        layer_fts = model(images[k: k+args.batch_size]).feature_maps
                        layer_fts = [layer_ft.unsqueeze(0) for layer_ft in layer_fts]
                        layer_fts = torch.cat(layer_fts, dim=0)
                        layer_fts = layer_fts.permute(1, 0, 2, 3, 4)
                        layer_fts = layer_fts.reshape(layer_fts.size(0), layer_fts.size(1), layer_fts.size(2), -1)
                        b_fts = layer_fts  # B, N, C, HW
                    b_fts = b_fts.data.cpu().numpy()
                    fts.append(b_fts)
                fts = np.concatenate(fts, 0)
                data = fts
                
                key = '%s_%s'%(scan_id, viewpoint_id)
                outf.create_dataset(key, data.shape, dtype='float32', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                
                num_finished_vps += 1
                progress_bar.update(num_finished_vps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ViT-B/32')
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--connectivity_dir', default='precompute_img_features/connectivity')
    parser.add_argument('--img_db', default='pretrain_src/img_features/habitat_256x256_vfov60_bgr.hdf5')
    parser.add_argument('--out_image_logits', action='store_true', default=False)
    parser.add_argument('--output_file', default='pretrain_src/img_features/Dinov2-views-habitat.hdf5')
    parser.add_argument('--batch_size', default=36, type=int)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--output_features', default=None, choices=['dino', 'clip'], required=True)
    args = parser.parse_args()

    build_feature_file(args)