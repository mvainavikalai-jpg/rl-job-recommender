# Reinforcement Learning for Job Role Recommendation

## 📌 Overview
A comparative study of Reinforcement Learning algorithms (Q-Learning, SARSA, and Deep Q-Network) to categorize candidates into job-fit levels (Entry, Mid, Senior) based on their resumes. The project highlights how Deep RL handles continuous state spaces better than traditional tabular methods.

## 🛠️ Tech Stack
- **Algorithms:** Q-Learning, SARSA, Deep Q-Network (DQN)
- **Libraries:** PyTorch, NumPy, Pandas, Scikit-learn
- **Dataset:** 30K AI Resume Screening Dataset (Kaggle)
- **Environment:** Google Colab (T4 GPU)

## 🚀 Methodology
1. **Environment Design:** Created a custom RL environment where the agent makes 20 sequential recommendations per episode. Class-balanced sampling was used to handle the minority "Entry Level" class (only 3.13% of data).
2. **State Representation:** 
   - *Tabular (Q-Learning/SARSA):* 6 integer bins (2,671 unique states).
   - *DQN:* 9 continuous float features (standardized + one-hot encoded).
3. **Reward System:** +1 for correct classification, -1 for incorrect. Shared across all agents.
4. **Training:** 5,000 episodes, γ=0.5, epsilon-greedy schedule (1.0 to 0.05). DQN used a target network and replay buffer for stability.

## 📊 Results
| Algorithm | Accuracy | Macro F1 | Entry F1 | Mean Reward |
| :--- | :--- | :--- | :--- | :--- |
| Q-Learning | 96.97% | 0.9269 | 0.8304 | 0.9393 |
| SARSA | 96.43% | 0.9082 | 0.7792 | 0.9287 |
| **DQN** | **99.70%** | **0.9923** | **0.9817** | **0.9940** |

## 💡 Key Insights
- **DQN Wins:** DQN outperformed tabular methods with ~90% fewer errors (18 errors vs 182/214). It learned exact thresholds (e.g., 2 yrs experience, 50 skills) instead of falling back to the majority class.
- **Tabular Limitation:** 160 unseen test states (2.67%) caused Q-Learning and SARSA to default to the "Mid Level" class, dropping their Macro F1 scores.
- **Accuracy vs Cost:** While tabular methods are fast, DQN's neural network function approximation is essential for handling continuous, high-dimensional candidate profiles.
