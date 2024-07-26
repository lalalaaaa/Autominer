# Autominer

## INTRODUCTION

As blockchain technology rapidly advances, it faces significant security threats from attacks like selfish mining, which exploit consensus algorithm vulnerabilities, undermining system security. Traditional analysis methods, primarily reliant on Markov models, are often utilized to address these complex threats. However, these methods frequently fall short in accurately simulating the multifaceted nature of blockchain attacks, leading to gaps in security measures. To bridge this gap, we introduce AutoMiner, an innovative reinforcement learning-based framework that integrates Miner Monte Carlo Tree Search (MMCTS) with Long Short-Term Memory (LSTM) networks for simulating and detecting potential mining attacks within the Proof of Work (PoW) framework. Our experimental results demonstrate AutoMiner's capability to outperform traditional selfish and honest mining strategies, achieving up to $20\%$ higher profits under specific scenarios. This highlights AutoMiner's potential in enhancing blockchain security by providing a more comprehensive and effective approach to analyzing and mitigating mining attacks.

![image](https://github.com/user-attachments/assets/2c70955e-c4cb-4f18-be15-d2679f36b7fe)
FIG: Illustrative overview of AutoMiner's operation and its strategic decision-making process.

## Highlights
Our key contributions are as follows:
* MMCTS Algorithm: We unveil the Miner Monte Carlo Tree Search (MMCTS), a pioneering simulation tool that deepens our understanding of mining strategies and their security implications. This algorithm not only broadens the scope of analysis but also enhances the precision of security assessments in blockchain networks.
* Neural Network Integration: By integrating LSTM neural networks, AutoMiner transcends traditional analysis methods, utilizing extensive historical data to predict mining behaviors with unmatched accuracy. This combination of machine learning and blockchain analytics sets a new standard in the field.
* Operational Efficiency and Adaptability: AutoMiner redefines efficiency, enabling complex simulations on conventional hardware while avoiding fixed attack patterns in favor of a dynamic exploration approach. Our strategy extends beyond Bitcoin, offering a robust framework for securing various blockchain architectures against evolving threats.

### Running simulation

#### Environment setup

Python 3.8 or higher.

```
git clone https://github.com/lalalaaaa/Autominer.git

cd Autominer

pip install -r requirements.txt
```

#### Run (to generate data)

```
python3 MCTS_4_Alpha_Miner.py
```

or

```
./MCTS_4_Alpha_Miner.py
```

#### Run (to train the model)

```
python3 LSTM.py
```

or

```
./LSTM.py
```
