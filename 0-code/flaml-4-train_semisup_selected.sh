#!/bin/bash

python flaml-4-train_semisupervised.py ./5-predictions/5-flaml-2D+fp_v2+embeddings+dyes_similarity+3D_XTB+3Dspectro_DWA_CLEAN/euos25_challenge_train_fluorescence480plus --strategy all --top-k 4 --confidence 0.95 --agreement 0.8 --iterations 5

python flaml-4-train_semisupervised.py ./5-predictions/5-flaml-2D+fp_v2+embeddings+dyes_similarity+3D_XTB+3Dspectro_DWA_CLEAN/euos25_challenge_train_transmittance450plus --strategy all --top-k 4 --confidence 0.95 --agreement 0.8 --iterations 5
  