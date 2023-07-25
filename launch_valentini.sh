#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

python train.py \
  dset=valentini \
  teacher_demucs.causal=1 \
  teacher_demucs.hidden=48 \
  teacher_demucs.resample=4 \
  student_demucs.causal=1 \
  student_demucs.hidden=16 \
  student_demucs.resample=4 \
  bandmask=0.2 \
  remix=1 \
  shift=8000 \
  shift_same=True \
  stft_loss=True \
  stft_sc_factor=0.1 stft_mag_factor=0.1 \
  segment=4.5 \
  stride=0.5 \
  ddp=1 $@


  

