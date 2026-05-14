# 《机器人学概论》课程大作业

## overview
大作业选题为基于课程强化学习的四足机器人低摩擦非平整地形自适应运动控制。  
采用宇树unitree_a1的机器狗。  
目前进展：环境创建+demo跑通
## 环境配置
```bash
conda create -n robot_proj python=3.10 -y
conda activate robot_proj
pip install mujoco gymnasium stable-baselines3 wandb moviepy
```
同时需要clone下来官方的宇树机器人库。
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```