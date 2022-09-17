#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATADIR=./data
LR=0.001
BS=128

printf '\n---------------------\n\n'

CORRUPT=snow
CUDA_VISIBLE_DEVICES=0 python TTAC_onepass_without_queue.py \
	--dataroot ${DATADIR} \
	--workers 4 \
	--corruption ${CORRUPT} \
	--batch_size ${BS} \
	--lr ${LR}