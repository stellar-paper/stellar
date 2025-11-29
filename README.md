# Replication package for the paper: "STELLAR: A Search-based Testing Framework for Large Language Model Applications"

## Overview
<p align="center">
  <img src="figures/approach-overview.png" alt="Architecture of STELLAR" width="400">
</p>

STELLAR is a modular search-based testing framework for evaluating LLM-based applications. It builds upon the <a href="https://www.github.com/opensbt">OpenSBT</a> infrastructure and provides the following core capabilities:

1. Integration of content, stylistic and perturbation features for test input generation & feature constraint handling
2. Automated test input generation using prompting and RAG-integration
3. Fitness evaluation leveraging LLM-based judgments alongside conventional similarity metrics (e.g., cosine similarity).
4. Search-based fitness optimization to support effectiv/efficient failure localization.

The results from the corresponding paper are provided partially in the repository as well as in the supplementary pdf in the links below.

## Links

Please refer to the following resources for detailed information:

[Go to STELLAR Tool / Replication Instructions](./stellar/)

[Go to Conversation Navi Tool](./naviqa/)

[Go to Detailed Results](./results/)

[Go to Supplementary Material](supplementary_material.pdf)
