# CarbonDynNet: Programmable Carbon Dynamic Aware Network

A novel computational framework for spatiotemporal evaluation and regulation of dynamic carbon transformation processes in Urban Drainage Networks (UDNs).

1. Overview

Dynamic carbon transformation within Urban Drainage Networks (UDNs) is a critical process for urban wastewater systems. UDNs act as distributed biogeochemical reactors, not just passive conduits. However, accurately evaluating and regulating these complex processes is extremely challenging due to their high spatiotemporal heterogeneity.

Traditional mechanistic models struggle to scale to complex network topologies, while standard machine learning (ML) models (like GNNs) lack mechanistic constraints, perform poorly with sparse data, and suffer from low interpretability.

CarbonDynNet (Programmable Carbon Dynamic Aware Network) solves this by embedding first-principle biokinetics (e.g., Monod equations) directly into the node information propagation mechanism of a Graph Neural Network (GNN).

2. Key Features

High Prediction Accuracy: CarbonDynNet significantly outperforms traditional ML models in simulating spatiotemporal carbon dynamics (R² = 0.95 ± 0.05 vs. R² = 0.11–0.87).

Robustness & Applicability: Demonstrates strong generalization and robustness across diverse network topologies, data noise, missing data, and sensor malfunctions (as shown in Fig. 3).

Mechanistic Interpretability: As a "knowledge-infused" or "physics-informed" model, CarbonDynNet can elucidate internal carbon transformation patterns and quantify the impact of key drivers (like hydraulic conditions).

Intelligent Regulation Framework: When combined with Reinforcement Learning (RL), the model serves as a "digital twin" to optimize flow rate control strategies in UDNs.

3. Framework and Methodology

Our framework (detailed in Fig. 1 of the manuscript) consists of three main stages:

Data Generation: A calibrated mechanistic model (SeweXnet) is used to generate a spatiotemporally complete dataset, overcoming the limitations of sparse real-world sensor coverage.

CarbonDynNet Modeling:

Topological Representation: The UDN is represented as a directed graph, with nodes as pipes and edges as hydraulic connections.

Mechanistic Embedding: Monod kinetics, environmental factor modulation, and a differentiable physics surrogate for particulate sedimentation are injected directly into the GCN message-passing layers.

Optimization & Regulation:

Key Node Identification: The NSGA-II algorithm is used to identify the most critical nodes for flow control.

Smart Regulation: A Reinforcement Learning (RL) agent is trained using CarbonDynNet as the environment. The agent learns an optimal flow rate control policy to minimize carbon loss and maximize the desired DOC fraction for downstream processes.

4. Key Results

Driver Identification: Hydraulic conditions (especially flow rate) are identified as the dominant drivers (46.7% of total impact) of carbon fraction heterogeneity (i.e., POC sedimentation and hydrolysis to DOC).

Hotspot Discovery: Upstream pipelines are revealed to be critical "hotspots" for carbon transformation due to sediment accumulation.

"Win-Win" Regulation: The RL-based control strategy successfully reduces total carbon loss from the UDN by 35.08 mg/L while simultaneously increasing the valuable DOC fraction by 12.71 mg/L.

Downstream Benefits: The optimized UDN effluent significantly improves the performance of downstream Wastewater Treatment Plants (WWTPs), enhancing denitrification and increasing potential biogas production by 27.5%.

5. Installation

(This is an example setup. Please modify based on your project's dependencies.)

# 1. Clone this repository
git clone [https://github.com/](https://github.com/)[YourUsername]/CarbonDynNet.git
cd CarbonDynNet

# 2. (Recommended) Create and activate a conda environment
conda create -n carbondyn python=3.8
conda activate carbondyn

# 3. Install dependencies
# (Assuming PyTorch and PyTorch Geometric)
# Find your specific CUDA/PyTorch command here: [https://pytorch.org/](https://pytorch.org/)
pip install torch torchvision torchaudio

# Install PyTorch Geometric (PyG)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f [https://data.pyg.org/whl/torch-$](https://data.pyg.org/whl/torch-$){TORCH}+${CUDA}.html

# Install other requirements
pip install -r requirements.txt


6. Usage / Quick Start

(This is an example. Please modify based on your code structure.)

a) Train the CarbonDynNet Model:

# Run the training script
python train.py --data_path /path/to/your/training_data.csv \
                --graph_path /path/to/your/graph_topology.adj \
                --model CarbonDynNet \
                --epochs 200


b) Run the RL-based Regulation:

# Load the pre-trained CarbonDynNet as the environment and run PPO
python run_rl_control.py --env_model_path /path/to/pretrained/carbondynnet.pth \
                         --node_list [8, 12, 25] \
                         --output_strategy /path/to/optimized_flow_strategy.csv


7. How to Cite

If you use this model or code in your research, please cite our paper:

(Please update this with the official publication details once available)

Yin, W.-X., Wang, Y.-Q., Wang, H.-C., Yan, Z., et al. (2025). Programmable Carbon Dynamic Aware Network for Spatiotemporal Evaluation and Regulation of Dynamic Carbon transformation process in Urban Drainage Network. 
8. Acknowledgements

This research was gratefully supported by:

Shenzhen Science and Technology Program (Grant Nos. JCYJ20241202123900001)

National Natural Science Foundation of China (No. 52321005)

9. Contact

Hong-Cheng Wang: wanghongcheng@hit.edu.cn

Ai-Jie Wang: waj0578@hit.edu.cn
