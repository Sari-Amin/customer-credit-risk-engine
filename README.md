# customer-credit-score-engine

## Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretability

The Basel II Accord emphasizes rigorous risk measurement through its pillars, particularly in the Internal Ratings-Based (IRB) approach. For a model to be used in calculating Risk-Weighted Assets (RWA), it must be interpretable, well-documented, and auditable by both internal teams and regulators. An interpretable model allows stakeholders to understand how inputs influence risk estimates, which is essential for transparency, capital adequacy assessments, and regulatory compliance. Clear documentation ensures reproducibility and provides the foundation for internal validation and supervisory review.

### 2. Why We Need a Proxy for Default — and Its Risks

In the absence of a clear default label (e.g., no “missed payment” field), we must define a proxy — often based on customer behavior (e.g., drop in transaction frequency or spending). This enables us to train supervised models but introduces several risks:

- **Misclassification**: A poor proxy might wrongly label good customers as defaulters or vice versa.
- **Business Loss**: Mislabeling can lead to rejected loans for creditworthy customers or risky loans to high-risk individuals.
- **Regulatory Risk**: If the proxy is not well-justified, regulators may reject the model.
- **Operational Overhead**: Additional effort is required to design, validate, and monitor proxy quality.

Using a proxy is often a necessity in emerging markets or alt-data scenarios, but it must be done with careful analysis and clear business reasoning.

### 3. Trade-offs: Simple vs Complex Models in Finance

| Criteria | Logistic Regression + WoE | Gradient Boosting |
|---------|----------------------------|-------------------|
| **Interpretability** | High — easy to explain to regulators and customers | Low — often a black box |
| **Regulatory Acceptance** | High | Medium to Low |
| **Accuracy** | Lower, especially on non-linear patterns | Higher, especially on large, complex data |
| **Feature Engineering** | Manual (e.g., WoE transformation) | Often automated |
| **Risk of Overfitting** | Lower | Higher — needs regularization |
| **Monitoring & Maintenance** | Easier | Harder |

In regulated financial environments, interpretability is typically prioritized. However, with tools like SHAP and LIME, the performance vs. transparency gap is narrowing, allowing a balance between accuracy and explainability.

![CI](https://github.com/Sari-Amin/customer-credit-score-engine/actions/workflows/ci.yml/badge.svg)

