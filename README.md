# Negatively Correlated Ensemble RL
Official code repository for paper "Negatively Correlated Ensemble Reinforcement Learning for Online Diverse Game Level Generation" in 2024 ICLR. The paper introduced an approach named negatively correlated ensemble reinforcement learning (NCERL), which is designed to tackle the issue of lacking diversity in RL-based real-time game level generation. To see the details, please read our [paper](https://openreview.net/).

If you used the code in this repository, please cite our paper:
```
@inproceedings{wang2024negatively,
  title={Negatively Correlated Ensemble Reinforcement Learning for Online Diverse Game Level Generation},
  author={Wang, Ziqi and Liu, Jialin},
  booktitle = {International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=iAW2EQXfwb},
}
```

### How to use

All training are launched by running `train.py` with option and arguments. For example, execute `python train.py ncesac --lbd 0.3 --m 5` will train NCERL with hyperparameters set as $\lambda = 0.3, m=5$.
 Plot script is `plots.py`

#### Options map to algorithms
* `python train.py gan`: to train a decoder which maps a continuous action to a game level segment.
* `python train.py sac`: to train a standard SAC as the policy for online game level generation
* `python train.py asyncsac`: to train a SAC with an asynchronous evaluation environment as the policy for online game level generation
* `python train.py ncesac`: to train an NCERL based on SAC as the policy for online game level generation
* `python train.py egsac`: to train an episodic generative SAC (see paper [*The fun facets of Mario: Multifaceted experience-driven PCG via reinforcement learning*](https://dl.acm.org/doi/abs/10.1145/3555858.3563282?casa_token=AHQWYSj_GyoAAAAA:MhwOltqfijP1NQj-c6NaTQikCnlNwyaMky07gCvTK5ZlSq063ew40awAcqEcw6S5zG9Sq9ZyDsspuaM)) as the policy for online game level generation
* `python train.py pmoe`: to train an episodic generative SAC (see paper [*Probabilistic Mixture-of-Experts for Efficient Deep Reinforcement Learning*](https://arxiv.org/abs/2104.09122)) as the policy for online game level generation
* `python train.py sunrise`: to train a SUNRISE (see paper [*SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep Reinforcement Learning*](https://proceedings.mlr.press/v139/lee21g.html)) as the policy for online game level generation
* `python train.py dvd`: to train a DvD-SAC (see paper [*Effective Diversity in Population Based Reinforcement Learning*](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d1dc3a8270a6f9394f88847d7f0050cf-Abstract.html)) as the policy for online game level generation

For the training arguments, please refer to the help `python train.py [option] --help`