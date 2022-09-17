#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATASET=cifar100

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

LR=0.001
BS=256


CUDA_VISIBLE_DEVICES=0 python TTAC_onepass2_without_queue.py \
	--dataroot ${DATADIR} \
	--dataset ${DATASET} \
	--resume results/${DATASET}_joint_resnet50 \
	--outf results/${DATASET}_ttac_no_without_queue \
	--corruption ${CORRUPT} \
	--level ${LEVEL} \
	--workers 4 \
	--batch_size ${BS} \
	--lr ${LR} \
	--num_sample ${NSAMPLE}
