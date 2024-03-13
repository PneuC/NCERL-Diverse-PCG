# Negatively Correlated Ensemble RL
Official code repository for paper "Negatively Correlated Ensemble Reinforcement Learning for Online Diverse Game Level Generation" in ICLR 2024. The paper introduced an approach named negatively correlated ensemble reinforcement learning (NCERL), which is designed to tackle the issue of lacking diversity in RL-based real-time game level generation. To see the details, please read our [paper](https://openreview.net/forum?id=iAW2EQXfwb).

If you used the code in this repository, please cite our paper:
```
@inproceedings{wang2024negatively,
  title={Negatively Correlated Ensemble Reinforcement Learning for Online Diverse Game Level Generation},
  author={Wang, Ziqi and Hu, Chengpeng and Liu, Jialin and Yao, Xin},
  booktitle = {International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=iAW2EQXfwb},
}
```

### Verified environment
* Python 3.9.6
* JPype 1.3.0
* dtw 1.4.0
* scipy 1.7.2
* torch 1.8.2+cu111
* numpy 1.20.3
* gym 0.21.0
* Pillow
* scipy 1.7.2
* Pillow 10.0.0
* matplotlib 3.6.3

### How to use

All training are launched by running `train.py` with option and arguments. For example, execute `python train.py ncesac --lbd 0.3 --m 5` will train NCERL with hyperparameters set as $\lambda = 0.3, m=5$.
 Plot script is `plots.py`

* `python train.py gan`: to train a decoder which maps a continuous action to a game level segment.
* `python train.py sac`: to train a standard SAC as the policy for online game level generation
* `python train.py asyncsac`: to train a SAC with an asynchronous evaluation environment as the policy for online game level generation
* `python train.py ncesac`: to train an NCERL based on SAC as the policy for online game level generation
* `python train.py egsac`: to train an episodic generative SAC (see paper [*The fun facets of Mario: Multifaceted experience-driven PCG via reinforcement learning*](https://dl.acm.org/doi/abs/10.1145/3555858.3563282?casa_token=AHQWYSj_GyoAAAAA:MhwOltqfijP1NQj-c6NaTQikCnlNwyaMky07gCvTK5ZlSq063ew40awAcqEcw6S5zG9Sq9ZyDsspuaM)) as the policy for online game level generation
* `python train.py pmoe`: to train an episodic generative SAC (see paper [*Probabilistic Mixture-of-Experts for Efficient Deep Reinforcement Learning*](https://arxiv.org/abs/2104.09122)) as the policy for online game level generation
* `python train.py sunrise`: to train a SUNRISE (see paper [*SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep Reinforcement Learning*](https://proceedings.mlr.press/v139/lee21g.html)) as the policy for online game level generation
* `python train.py dvd`: to train a DvD-SAC (see paper [*Effective Diversity in Population Based Reinforcement Learning*](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d1dc3a8270a6f9394f88847d7f0050cf-Abstract.html)) as the policy for online game level generation

For the training arguments, please refer to the help `python train.py [option] --help`


#### Some related works
* Z. Wang, C. Hu, J. Liu and X. Yao, "[The fun facets of Mario: Multifaceted experience-driven PCG via reinforcement learning](https://dl.acm.org/doi/abs/10.1145/3555858.3563282?casa_token=AHQWYSj_GyoAAAAA:MhwOltqfijP1NQj-c6NaTQikCnlNwyaMky07gCvTK5ZlSq063ew40awAcqEcw6S5zG9Sq9ZyDsspuaM)," in *Proceedings of the 17th International Conference on the Foundations of Digital Games*, 2022, pp. 1-8
* T. Shu, J. Liu and G. N. Yannakakis, "[Experience-Driven PCG via Reinforcement Learning: A Super Mario Bros Study](https://ieeexplore.ieee.org/document/9619124)," in *2021 IEEE Conference on Games*, 2021, pp. 1-9