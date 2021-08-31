# STOI-Net: A Deep Learning based Non-Intrusive Speech Intelligibility Assessment Model

**Introduction**

The calculation of most objective speech intelligibility assessment metrics requires clean speech as a reference. Such a requirement may limit the applicability of these metrics in real-world scenarios. To overcome this limitation, we propose a deep learning-based non-intrusive speech intelligibility assessment model, namely STOI-Net. The input and output of STOI-Net are speech spectral features and predicted STOI scores, respectively. The model is formed by the combination of a convolutional neural network and bidirectional long short-term memory (CNN-BLSTM) architecture with a multiplicative attention mechanism. Experimental results show that the STOI score estimated by STOI-Net has a good correlation with the actual STOI score when tested with noisy and enhanced speech utterances. The correlation values are 0.97 and 0.83, respectively, for the seen test condition (the test speakers and noise types are involved in the training set) and the unseen test condition (the test speakers and noise types are not involved in the training set). The results confirm the capability of STOI-Net to accurately predict the STOI scores without referring to clean speech.

For more detail please check our <a href="https://arxiv.org/ftp/arxiv/papers/2011/2011.04292.pdf" target="_blank">Paper</a>

**Installation**

You can download our environmental setup at Environment Folder and use the following script.
```js
conda env create -f environment.yml
```

**Citation**

Please kindly cite our paper, if you find this code is useful.

<a id="1"></a> 
R. E. Zezario, S.-W. Fu, C.-S. Fuh, Y. Tsao, and H.-M. Wang, “STOINet: A deep learning based non-intrusive speech intelligibility assessment model,” in Proc. APSIPA ASC, 2020

**Note**

<a href="https://github.com/CyberZHG/keras-self-attention" target="_blank">Self Attention</a> function is created by others
