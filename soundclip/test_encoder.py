from librosa.core import audio
from scipy.signal import waveforms
from torch.utils.data.dataset import Dataset
import librosa
from glob import glob
import cv2
import numpy as np
import math
import torch
import random
import os
import pandas as pd
import time
from datasets import VggsoundCurationDataset, AudiosetBalancedCurationDataset, AudiosetUnbalancedCurationDataset
from models import AudioEncoder
import clip
from torch.nn import CosineSimilarity
# from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel

cossim = CosineSimilarity(dim=0, eps=1e-6)


def dist(v1, v2):
   return cossim(v1, v2)


wav_dir = "../audiosample/"
wav_names = os.listdir(wav_dir)
n_mels = 128
time_length = 864
resize_resolution = 512
audio_inputs = []
text_prompts = []
for wav_name in wav_names:
   text_prompt = wav_name.split("/")[-1].split(".")[0]
   y, sr = librosa.load(wav_dir+wav_name, sr=44100)
   audio_input = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
   audio_input = librosa.power_to_db(audio_input, ref=np.max) / 80.0 + 1

   zero = np.zeros((n_mels, time_length))
   resize_resolution = 512
   h, w = audio_input.shape
   if w >= time_length:
      j = 0
      j = random.randint(0, w-time_length)
      audio_input = audio_input[:,j:j+time_length]
   else:
      zero[:,:w] = audio_input[:,:w]
      audio_input = zero
      audio_input = cv2.resize(audio_input, (n_mels, resize_resolution))
   audio_input = audio_input.reshape(-1, n_mels, resize_resolution)
   audio_input = torch.from_numpy(audio_input).float()
   audio_inputs.append(audio_input)
   text_prompts.append(text_prompt)


pretrained_path = "../pretrained_models/resnet18_57.pth"
audioencoder = AudioEncoder()
audioencoder.load_state_dict(torch.load(pretrained_path))
audioencoder = audioencoder.cuda()
audioencoder.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/32", device=device)
ce = torch.nn.CrossEntropyLoss()

with torch.no_grad():
   audio_inputs = torch.cat(audio_inputs, dim=0).reshape(-1,1,n_mels,resize_resolution)
   audio_embedding = audioencoder(audio_inputs.cuda())
   # audio_aug_embedding = audioencoder(audio_inputs.cuda())

   text_tokens = torch.cat([clip.tokenize(text) for text in text_prompts])
   text_embedding = clip_model.encode_text(text_tokens.to(device)).float()
   text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

   audio_embedding = audio_embedding / audio_embedding.norm(dim=-1, keepdim=True)
   # audio_aug_embedding = audio_aug_embedding / audio_aug_embedding.norm(dim=-1, keepdim=True)

   loss = 0

   projection_audio_text = (audio_embedding @ text_embedding.T) * math.exp(0.07)
   # projection_self_audio = (audio_embedding @ audio_aug_embedding.T) * math.exp(0.07)

   cos_sim_matrix = torch.zeros_like(projection_audio_text)
   print(text_prompts)
   print(projection_audio_text)
   # label = torch.arange(args.batch_size, dtype=torch.long).cuda()

   # audio_contrastive_loss = ce(projection_audio_text, label) + ce(projection_audio_text.T, label)
   # self_contrastive_loss = ce(projection_self_audio, label) + ce(projection_self_audio.T, label)
   # loss = (audio_contrastive_loss + self_contrastive_loss) / 4