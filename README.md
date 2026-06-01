# Neuro-Symbolic Link Prediction on the Twitch Social Network

Link prediction determines whether a connection is likely to exist between two nodes in a network. This project applies a neuro-symbolic framework to predict friendship links in the Twitch English social network — a graph of 7,126 users and 35,324 mutual connections.

A logistic regression baseline achieved strong overall accuracy (84%) but missed 25% of actual links (recall = 0.75). Error analysis revealed the model consistently failed on connected users who share no mutual friends. A symbolic rule derived from **preferential attachment theory** was encoded as a training loss penalty to directly target this failure, reducing missed connections by 26%.

## Results

| System | Accuracy | AUC | Precision | Recall | F1 | False Negatives |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.84 | 0.91 | 0.92 | 0.75 | 0.83 | 1,747 |
| MLP (BCE only) | 0.85 | 0.92 | 0.90 | 0.80 | 0.84 | 1,439 |
| MLP + Rule (λ=0.5) | 0.85 | 0.92 | 0.89 | 0.81 | 0.85 | 1,343 |
| MLP + Rule (λ=1.0) | **0.85** | **0.92** | 0.88 | **0.82** | **0.85** | **1,284** |
| MLP + Rule (λ=2.0) | 0.85 | 0.92 | 0.88 | 0.82 | 0.85 | 1,263 |

The symbolic rule recovered 463 real connections (-26% false negatives) that the LR baseline missed, with λ=1.0 as the preferred setting.

## Approach

This project follows a **Neuro: Symbolic→Neuro** design from [Kautz's taxonomy](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/18229) of neuro-symbolic AI systems. Symbolic knowledge is used during training to shape the network's weights; at inference time the network alone makes predictions.

**Symbolic rule:** preferential attachment theory states that nodes with higher degree are more likely to form new connections. If two Twitch users are connected, share no mutual friends (Adamic-Adar = 0), and are both sufficiently popular (preferential attachment > 30), the model should not confidently predict non-link:

$$(\text{label} = 1) \wedge (\text{AA} = 0) \wedge (\text{PA} > 30) \Rightarrow \hat{y} = 1$$

This rule is enforced via a penalty term added to the BCE loss during training:

$$\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda \cdot \frac{1}{|M|} \sum_{i \in M} \max(0,\ 0.4 - p_i)$$

where $M$ is the set of training samples satisfying the rule and $\lambda$ controls enforcement strength. The penalty fires only when the model predicts less than 40% link probability for a pair that satisfies the rule — if the model is already confident enough, training is unaffected.

This is related to [Semantic Loss](https://proceedings.mlr.press/v80/xu18h.html), which similarly encodes logical constraints as differentiable penalty terms.

## Project Structure

```
├── data/
│   ├── musae_ENGB_edges.csv              # Raw graph edges
│   ├── musae_ENGB_target.csv             # Node attributes (views, age, partner, mature)
│   └── twitch.csv                        # Final dataset (9 features per node pair)
├── notebooks/
│   ├── link_prediction_build_dataset.ipynb  # Build dataset from raw inputs (PySpark)
│   ├── link_prediction_baseline.ipynb       # Logistic regression baseline
│   ├── link_prediction_nesy.ipynb           # MLP baseline + neuro-symbolic MLP
│   └── output/
│       ├── README.md                        # Full experiment results
│       ├── 20260524_2044_lr/
│       ├── 20260524_2044_mlp_bce/
│       ├── 20260524_2049_mlp_lambda0.5/
│       ├── 20260524_2052_mlp_lambda1.0/
│       └── 20260524_2055_mlp_lambda2.0/
├── report/
│   ├── neuro_symbolic_link_prediction.md   # Written report (source)
│   ├── neuro_symbolic_link_prediction.pdf  # Written report (PDF)
│   └── build_pdf.py                        # Regenerate PDF from markdown
└── pyproject.toml
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

Or with pip directly:

```bash
pip install pandas numpy scikit-learn torch matplotlib seaborn jupyter
```

Python 3.13+ required.

## Reproducing Results

`data/twitch.csv` is included in the repository — no rebuild needed to run the notebooks.

```bash
# LR baseline
jupyter notebook notebooks/link_prediction_baseline.ipynb

# MLP baseline and neuro-symbolic MLP
jupyter notebook notebooks/link_prediction_nesy.ipynb
```

In `link_prediction_nesy.ipynb`, toggle the flags at the top of the training cell:

```python
USE_SYMBOLIC_LOSS = True   # False for plain MLP baseline
LAMBDA            = 1.0    # 0.5, 1.0, or 2.0
```

Set these to reproduce each row in the results table.

## Dataset

The [Twitch English social network](https://snap.stanford.edu/data/twitch-social-networks.html) from the MuSAE dataset: 7,126 nodes (streamers) and 35,324 edges (mutual follows). The classification dataset contains 70,648 node pairs — balanced between true links and randomly sampled non-links. Each pair is described by 9 features: 4 structural graph features (common neighbors, Jaccard coefficient, Adamic-Adar, preferential attachment) and 5 pairwise node attribute features (view count difference/ratio, account age difference, same partner status, same mature-content flag).

## References

1. Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509–512.
2. Kautz, H. (2022). The third AI summer. *AI Magazine*, 43(1), 93–104.
3. Rozemberczki, B., Allen, C., & Sarkar, R. (2019). Multi-scale attributed node embedding. arXiv:1909.13021.
4. Xu, J., Zhang, Z., Friedman, T., Liang, Y., & Van den Broeck, G. (2018). A semantic loss function for deep learning with symbolic knowledge. *ICML 2018*.
