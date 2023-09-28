from src.env.rfunc import *

default = lambda : RewardFunc(LevelSACN(), GameplaySACN(), Playability())
fp = lambda : RewardFunc(Fun(200), Playability(10))
hp = lambda : RewardFunc(HistoricalDeviation(1), Playability(1))
fhp = lambda : RewardFunc(Fun(30, num_windows=21), HistoricalDeviation(3), Playability(3))
lp = lambda : RewardFunc(LevelSACN(2.), Playability(2.))
gp = lambda : RewardFunc(GameplaySACN(2.), Playability(2.))
lgp = lambda : RewardFunc(LevelSACN(), GameplaySACN(), Playability())
