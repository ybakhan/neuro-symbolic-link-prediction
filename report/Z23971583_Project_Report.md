**Name:** Yasser Khan  
**Z-Number:** Z23971583  
**Course:** COT 6930 Project Report

# Neuro-Symbolic Link Prediction on the Twitch Social Network

### 1. Introduction

Link prediction is the task of determining whether a connection is likely to exist between two nodes in a network. In social networks such as Twitch, link prediction enables friend recommendation by identifying pairs of users who are likely to connect. This report describes the application of a neuro-symbolic framework to link prediction on the Twitch English social network [3], a dataset of 7,126 users and 35,324 mutual connections.

A logistic regression (LR) baseline achieved 84% accuracy but missed 25% of actual links (recall = 0.75). Closer inspection of the errors showed the model learned to predict a link only when two users share at least one mutual friend (Adamic-Adar score > 0). Pairs of connected users who share no mutual friends are always predicted as non-link. This report addresses that failure by replacing the LR model with a neuro-symbolic multi-layer perceptron (MLP) that incorporates domain knowledge about how popular users tend to connect even without shared friends.

### 2. Dataset

The dataset consists of 70,648 pairs of Twitch users derived from the MuSAE Twitch English network [3]. Each pair is labeled 1 if the two users are mutually connected and 0 if they are not. The dataset is balanced, with 35,324 samples of each class. The non-link pairs were randomly sampled from all non-connected node pairs in the network. Each pair is described by 9 features listed in Table 1. An 80/20 train/test split was used with 10-fold cross-validation on the training set.

| Feature | Description |
|---|---|
| common_neighbors | Count of mutual connections |
| jaccard_coefficient | Neighbor overlap relative to neighbor union |
| adamic_adar (AA) | Common neighbors weighted by their degree |
| preferential_attachment (PA) | Product of the two nodes' degrees |
| views_diff | Absolute difference in total view counts |
| views_ratio | min(views) / max(views) relative popularity |
| age_diff | Absolute difference in account age (days) |
| same_partner | 1 if both or neither users are Twitch partners |
| same_mature | 1 if both or neither users have their channel flagged as mature content |

Table 1. Features computed for each pair of Twitch users.

### 3. Neuro-Symbolic Framework

According to Kautz's taxonomy of neuro-symbolic designs [2], the proposed approach follows a **Neuro: Symbolic→Neuro** design. During training, a symbolic rule is used to compute an additional loss term that drives the network's weight updates. At inference time, the network alone makes predictions. The neural component is an MLP with two hidden layers of 64 and 32 nodes (ReLU activations) trained with binary cross-entropy (BCE).

The symbolic component is derived from preferential attachment theory, which states that nodes with higher degree are more likely to form new connections [1]. In the Twitch network, if two users are connected, share no mutual friends (AA = 0), and are both sufficiently popular (PA > 30), the model should not confidently predict non-link. The threshold PA > 30 was selected based on the LR false negatives (FNs), where 93.1% had AA = 0 and a PA median of 88. Equation 1 expresses the symbolic rule as a propositional formula.

$$(\text{label} = 1) \wedge (\text{AA} = 0) \wedge (\text{PA} > 30) \Rightarrow \hat{y} = 1$$

Equation 1. Symbolic rule expressed as a propositional formula.

To enforce this rule during training, a penalty term is added to the BCE loss. The penalty fires only when the model predicts less than 40% probability of a link for a pair that satisfies the rule. The penalty grows as the predicted probability is further below that threshold. If the model is confident enough, the penalty is zero and training is unaffected. Equation 2 shows the BCE loss augmented with this penalty term. $M$ is the set of training samples satisfying the rule, $p_i$ is the predicted link probability, and $\lambda$ is a hyperparameter that controls how strongly the rule is enforced during training.

$$\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda \cdot \frac{1}{|M|} \sum_{i \in M} \max(0,\ 0.4 - p_i)$$

Equation 2. BCE loss augmented with a symbolic penalty term.

### 4. Experiments

Three values of $\lambda$ were evaluated on the test set: 0.5, 1.0, and 2.0. Table 2 shows the evaluation metrics across all systems.

| System | Accuracy | AUC | Precision | Recall | F1 | FN | FP |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.84 | 0.91 | 0.92 | 0.75 | 0.83 | 1,747 | 471 |
| MLP (BCE only) | 0.85 | 0.92 | 0.90 | 0.80 | 0.84 | 1,439 | 649 |
| MLP + Rule (λ=0.5) | 0.85 | 0.92 | 0.89 | 0.81 | 0.85 | 1,343 | 721 |
| MLP + Rule (λ=1.0) | 0.85 | 0.92 | 0.88 | **0.82** | **0.85** | **1,284** | 777 |
| MLP + Rule (λ=2.0) | 0.85 | 0.92 | 0.88 | **0.82** | **0.85** | **1,263** | 806 |

Table 2. Evaluation metrics across all systems. FN = false negatives, FP = false positives.

Moving from LR to plain MLP reduces FNs by 308 (-18%), as the non-linear decision boundary captures patterns LR cannot. Adding the symbolic rule to the MLP at λ=1.0 reduces FNs by a further 155 (-11%). These are specifically the AA=0 user pairs that the rule targets. The neuro-symbolic system recovers 463 real connections (-26%) that the LR baseline missed. The λ=1.0 setting is preferred over λ=2.0 since both achieve identical recall and F1, but λ=1.0 produces 29 fewer FPs (777 vs 806).

### 5. Discussion

In a friend recommendation system, a false negative means two users who could have been friends are never suggested to each other. A false positive means a suggestion is made that the user may not accept, which is harmless. This makes recall the more important metric, and justifies accepting more FPs in exchange for fewer missed connections.

The symbolic loss encodes a proposition from network science into the training process as a logical rule. This is different from adjusting a classification threshold, which affects all samples uniformly. The symbolic loss only penalizes user pairs that satisfy the rule and are predicted as non-link. This is related to Semantic Loss [4], which similarly encodes logical constraints as penalty terms added to the training loss.

The system has several limitations. 95% of remaining FNs at λ=1.0 still have AA = 0, meaning pairs with no shared friends and low popularity are indistinguishable from true non-links. Negative samples were drawn at random rather than from nearby nodes in the graph. Most randomly sampled pairs are far apart in the Twitch network and easy to classify, which inflates model performance. Sampling non-connected pairs at distance 2 or 3 hops would test the model on harder cases and provide a more realistic evaluation. The penalty threshold of 0.4 means user pairs predicted between 0.4 and 0.5 are classified as non-link by the model but are not penalized by the rule. This is a trade-off between training stability and catching more missed connections.

### 6. Conclusion

This report presented a neuro-symbolic approach to link prediction on the Twitch social network. The LR baseline missed real connections between users who share no mutual friends. A symbolic rule derived from preferential attachment theory was added to the MLP loss function to target this failure. The system reduced FNs by 463 (-26%) and improved link recall from 0.75 to 0.82.

### References

[1] Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. _Science_, 286(5439), 509–512.

[2] Kautz, H. (2022). The third AI summer. _AI Magazine_, 43(1), 93-104.

[3] Rozemberczki, B., Allen, C., & Sarkar, R. (2019). Multi-scale attributed node embedding. arXiv:1909.13021. Dataset: https://snap.stanford.edu/data/twitch-social-networks.html

[4] Xu, J., Zhang, Z., Friedman, T., Liang, Y., & Van den Broeck, G. (2018). A semantic loss function for deep learning with symbolic knowledge. _ICML 2018_.
