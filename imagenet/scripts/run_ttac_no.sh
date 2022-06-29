#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

DATADIR=./data
LR=0.0001
BS=128

printf '\n---------------------\n\n'

CORRUPT=snow
CUDA_VISIBLE_DEVICES=7 python TTAC_onepass.py \
	--dataroot ${DATADIR} \
	--workers 4 \
	--iters 2 \
	--corruption ${CORRUPT} \
	--batch_size ${BS} \
	--lr ${LR}