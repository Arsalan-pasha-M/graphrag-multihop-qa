# Evaluation (n=30)

|                       |   baseline_rag |   graph_rag |
|:----------------------|---------------:|------------:|
| EM                    |          0.433 |       0.467 |
| F1                    |          0.482 |       0.554 |
| support_recall        |          0.717 |       0.75  |
| full_support          |          0.467 |       0.533 |
| full_support_evidence |          0.467 |       0.7   |
| citation_precision    |          0.883 |       0.817 |
| avg_latency_s         |          6.863 |      28.17  |
| p95_latency_s         |          9.735 |      71.03  |
| invalid_json_rate     |          0     |       0     |
| avg_context_words     |        443.733 |     792.567 |

## By difficulty

|                          |    em |    f1 |   full_support |
|:-------------------------|------:|------:|---------------:|
| ('baseline_rag', 'hard') | 0.433 | 0.482 |          0.467 |
| ('graph_rag', 'hard')    | 0.467 | 0.554 |          0.533 |
