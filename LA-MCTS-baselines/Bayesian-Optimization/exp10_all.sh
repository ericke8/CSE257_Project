#!/bin/bash

for i in {1..5}
do
    # python run.py --func ackley --dims 10 --iterations 300 >> ackley_res.log
    # python run.py --func levy --dims 10 --iterations 300 >> levy_res.log
    # python run.py --func schwefel --dims 10 --iterations 300 >> schwefel_res.log
    python run.py --func minigrid --dims 21 --iterations 300 >> minigrid_res.log
done