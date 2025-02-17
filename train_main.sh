#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
name="experiment"
python3 mainclean.py --name $name --stateful --batch-size 10 --slowsteps 6 --faststeps 6 