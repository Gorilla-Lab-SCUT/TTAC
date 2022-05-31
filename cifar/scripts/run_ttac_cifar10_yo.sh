#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATASET=cifar10

DATADIR=./data

# ===================================

LEVEL=5

if [ "$#" -lt 2 ]; then
	CORRUPT=snow
	NSAMPLE=100000
else
	CORRUPT=$1
	NSAMPLE=$2
fi

# ===================================

LR=0.01
BS_SSL=256
BS_ALIGN=256


CUDA_VISIBLE_DEVICES=0 python TTAC_onepass.py \
	--dataroot ${DATADIR} \
	--dataset ${DATASET} \
	--resume results/${DATASET}_joint_resnet50 \
	--outf results/${DATASET}_ttac_yo \
	--corruption ${CORRUPT} \
	--level ${LEVEL} \
	--workers 4 \
	--batch_size ${BS_SSL} \
	--batch_size_align ${BS_ALIGN} \
	--lr ${LR} \
	--num_sample ${NSAMPLE} \
	--iters 4 \
	--align_ext \
	--with_ssl
