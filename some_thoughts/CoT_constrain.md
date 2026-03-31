问题：在做Omni模型，之前是想通过SFT把输出格式化，分别输出视频和音频模态的推理后，再整体推理输出答案。并且在视频和音频模态的推理里SFT出来细粒度信息，比如表情、音高之类的

pipeline 理解：
step1: 输入：video + audio + question -> model， 输出：video_response， audio_response （fine-grained）
step2: 输入： video_response + audio_response -> model， 输出： final_response

目标：需要有一个约束CoT阶段单模态信息有效性的创新
目标理解：目前step1想要通过SFT来实现，目前想要通过一种新的足够novel的方法来构建CoT输出，保证step2能够有效利用视频和音频模态的推理信息。

整体论文的故事核心在于构建一个合理且足够novel的RL pipeline

## 一些是否已经思考或者检查过的点
1. 现有方法是否已经能够有效利用单模态信息 -》 寻找现有方法的缺点，作为改进的核心方向
2. 视频模态OOM -》 现有方法是否存在该问题，以及是否能够成为核心contribution
3. 目前对于你们的pipeline而言，video和audio输入是混合状态还是分离状态（我的理解是和Qwen3-Omni保持一致，混合）

## 后续可能的思路
1. 简单看了一些paper，都是基于GRPO算法做改进，调整一些细节（loss设计或者多阶段优化训练）
2. 我之前做过多阶段的GRPO算法改进，思路很简单（调整数据格式和loss设计），但是效果不错，希望有所帮助：https://arxiv.org/pdf/2510.21311

## 一些尚有疑问的点
1. 你们的对比方法主要有哪些，例如Qwen3-Omni或者一些类似的模型，差距目前多大；
2. 核心对比的task有哪些，benchmark

## 个人建议
先以实际效果为准，可以先考虑把算法结果实现出来，然后转化为contribution

