# RLexperiments
This project contains various reinforcement/imitation learning experiments, such as implementations of Implicit Quantile Networks \[1\], Neural Episodic Control \[2\], FeUdal Networks \[3\], Bootstrapped Dual Policy Iteration \[4\], Prioritized Experience Replay \[5\], TD3 \[6\]. Most of them are applied to the MineRL enviroment package. The two latest experiments, an off-policy multiaction actor-critic agent and an Upside Down RL agent \[7\], are featured in the root directory of this repository. One resulting multiaction agent can be seen here https://www.youtube.com/watch?v=oyNKCeMywtY. The other experiments can be found in the all_experiments/ directory.

This project also features an easily extendable multi-processing system of producer/consumer workers (trajectory generator, replay buffer and trainer workers).

This repository contains modified and unmodified code from:
- https://github.com/facebookresearch/XLM (lightly modified HashingMemory module)
- https://github.com/minerllabs/minerl (modified data pipeline)
- https://github.com/minerllabs/baselines (modified and new wrappers)
- https://github.com/LiyuanLucasLiu/RAdam
- https://github.com/pytorch/pytorch (modified MixtureSameFamily)
- https://github.com/dannysdeng/dqn-pytorch (modified IQN)
- https://github.com/zxkyjimmy/Lookahead
- https://github.com/kazizzad/GATS (for spectral norm)
- https://github.com/christiancosgrove/pytorch-spectral-normalization-gan (for spectral norm)
- https://github.com/rlcode/per (for the SumTree class)
- https://github.com/vub-ai-lab/bdpi (Learner, Actor, Critic and LearnerModel class structure are used as templates)

References:

\[1\] W. Dabney et al. Implicit quantile networks for distributional reinforcement learning. In International Conference on Machine Learning, pages 1104–1113, 2018.

\[2\] A. Pritzel et al. Neural episodic control. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 2827–2836. JMLR. org, 2017.

\[3\] A. S. Vezhnevets et al. Feudal networks for hierarchical reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 3540–3549. JMLR. org, 2017.

\[4\] D. Steckelmacher et al. Sample-efficient model-free reinforcement learning with off-policy critics. arXiv preprint arXiv:1903.04193, 2019.

\[5\] S. Tom et al. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.

\[6\] S. Fujimoto et al. Addressing Function Approximation Error in Actor-Critic Methods. International Conference on Machine Learning. 2018.

\[7\] J. Schmidhuber. Reinforcement Learning Upside Down: Don't Predict Rewards--Just Map Them to Actions. arXiv preprint arXiv:1912.02875, 2019.
