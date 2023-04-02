# Min Max Actor Critic Learning Automation

Here is an implementation of min max actor critic learning automation, modified to solve classification problems. I've also used the same algorithm for solving classic control problems like cartpole and mountain car. Algorithm is taken from [this](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/rl_classification.pdf) paper. They've proposed a methodology to transform classification problem to reinforcement learning problem and in addition to that they've also proposed a modified version of an RL algorithm (min max actor critic learning automation) for this job.

I've tested the algorithm on [Titanic](https://www.kaggle.com/c/titanic/data) and [Diabetes](https://www.kaggle.com/datasets/saurabh00007/diabetescsv) data for classification. The data was taken from kaggle. And for classic control problems I used Mountain car and Cartpole environments. I haven't provided the data and trained models in this repo.

## Folder structure

rl_classification
├── algorithms
│   ├── m2acla.py
│   └── __pycache__
├── cart-pole
│   ├── cart-pole.py
│   └── rewards-plot.png
├── data
│   ├── diabetes
│   └── titanic
├── diabetes
│   ├── diabetes.py
│   ├── roc-curve.png
│   └── score-dist.png
├── environments
│   ├── environments.py
│   └── __pycache__
├── models
│   ├── cart-pole
│   ├── diabetes
│   ├── mountain-car
│   └── titanic
├── mountain-car
│   ├── mountain-car.py
│   └── rewards-plot.png
├── readme.md
├── requirements.txt
└── titanic
    ├── roc-curve.png
    ├── score-dist.png
    └── titanic.py

