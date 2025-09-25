# FTRL

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

这是一份 FTRL 和 FTRL-Adam 的 PyTorch 实现，主要用于原理说明。该项目是 [这篇博客](https://blog.yuime.moe/posts/online-learning/) 的随附代码。

该项目是一份可用的代码，但不是一份优秀的代码，包含了许多问题：没有参数检查、大量重复的代码、低下的执行效率、缺乏文档和注释……不过它至少保证了 3M 原则的 work 和 right。

## 安装

直接复制 [ftrl.py](./ftrl.py) 就好，单文件模块，不过真要说算得上安装指令的应该是这一条：

```bash
git clone https://github.com/ppq1024/FTRL.git
```

## 使用方法

两个核心类继承自 `torch.optim.Optimizer`，像其它优化器一样使用就行，但不保证 PyTorch 的部分特性正常工作。

[ftrl.ipynb](./ftrl.ipynb) 包含了测试代码和结果，可以参考一下（model 随手捏的别吐槽）。

## 测试环境

测试过程中的关键包版本如下：

- python 3.13.2
- pytorch 2.6.0+cu126
- torchvision 0.21.0+cu126

## 如何贡献

欢迎使用 [gitalk](https://github.com/gitalk/gitalk) 在 [原博客](https://blog.yuime.moe/posts/online-learning/#gitalk-container) 下进行评论，直接 [提一个 Issue](https://github.com/ppq1024/FTRL/issues/new) 或者提交一个 Pull Request 也是不错的选择。

## 使用许可

[MIT](LICENSE) © IRID, PPQ
