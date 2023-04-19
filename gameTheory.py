#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:33:56 2023

@author: chrisfeng
"""

import pybaseball as pb
import pandas as pd
import numpy as np
import nashpy as nash
import matplotlib.pyplot as plt

batterName = "Aaron Judge"
pitcherName = "Patrick Corbin"

bName = batterName.split()
pName = pitcherName.split()

batterNum = pb.playerid_lookup(bName[1],bName[0]).iloc[0,2]
pitcherNum = pb.playerid_lookup(pName[1],pName[0]).iloc[0,2]

pitcher = pb.statcast_pitcher('2020-04-01', '2023-04-15', player_id = pitcherNum)
batter = pb.statcast_batter('2020-04-01', '2023-04-15', player_id = batterNum)

dropIDX = pitcher[pitcher['pitch_name']=="Slow Curve"].index
pitcher.drop(dropIDX, inplace=True)

pDeltaExp = pitcher.groupby(["pitch_name"]).delta_run_exp.agg(['mean'])
bDeltaExp = batter.groupby(["pitch_name"]).delta_run_exp.agg(['mean'])

#Pitch distribution
pitchDist = np.array(pitcher.groupby(["pitch_name"]).pitch_name.agg(['count']))
pitchDist = (pitchDist/sum(pitchDist)).T

#Row number to pitches
pitchIndex = pDeltaExp.index.values

#Match batter and pitcher
bDeltaExp = np.array(bDeltaExp.loc[pDeltaExp.index.values,:])
pDeltaExp = np.array(pDeltaExp)

#Expanding the matrix to a square
bDeltaExp = np.tile(bDeltaExp,(1,bDeltaExp.shape[0]))
pDeltaExp = np.tile(pDeltaExp,(1,bDeltaExp.shape[0]))

#Creating Payoff Matrix
payoffs = bDeltaExp+pDeltaExp

diag = payoffs[np.diag_indices_from(payoffs)]

payoffs = payoffs-(0.25)*abs(payoffs)

payoffs[np.diag_indices_from(payoffs)] = diag\
    +0.5*abs(diag*\
              pitchDist)

#Zero-sum assumption -> row player is batter, column player is hitter
zeroGame = nash.Game(payoffs)
np.random.seed(369)
iterations = 5000
play_counts_and_distributions = tuple(zeroGame.stochastic_fictitious_play(iterations=iterations))

#Equilibrium
equilibria = zeroGame.support_enumeration()
eqDf = pd.DataFrame({"Pitch Type":pitchIndex})
for eq in equilibria:
    eqDf.insert(1, "Batter", eq[0])
    eqDf.insert(1, "Pitcher", eq[1])
    print(eqDf)


#Stochastic Sim
for play_counts, distributions in play_counts_and_distributions:
    batCount, pitchCount = play_counts
    batDist, pitchDist = distributions

results = pd.DataFrame({"Pitch Type":pitchIndex, "Batters Distribution":batDist, \
                        "Pitcher's Distribution":pitchDist})
print(results)

#Graph Convergence for simulated game
plt.figure() 
probabilities = [
    batCount / np.sum(batCount)
    if np.sum(batCount) != 0
    else batCount + 1 / len(batCount)
    for (batCount, pitchCount), _ in play_counts_and_distributions]
for number, strategy in enumerate(zip(*probabilities)):
    plt.plot(strategy, label=f"$s_{number}$") 
plt.xlabel("Iteration") 
plt.ylabel("Probability") 
plt.title("Actions taken by row player") 
plt.legend() 












