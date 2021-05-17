#!/bin/bash

for i in {1..5}
do
    python run.py --func ackley --dims 10 --iterations 200 >> ackley_mcts_res.log
    python run.py --func levy --dims 10 --iterations 200 >> levy_mcts_res.log
    python run.py --func schwefel --dims 10 --iterations 200 >> schwefel_mcts_res.log
done