# Evaluation (n=20)

|                       |   baseline_rag |   graph_rag |
|:----------------------|---------------:|------------:|
| EM                    |          0.4   |       0.5   |
| F1                    |          0.525 |       0.691 |
| support_recall        |          0.85  |       0.875 |
| full_support          |          0.7   |       0.75  |
| full_support_evidence |          0.7   |       0.9   |
| citation_precision    |          0.925 |       0.9   |
| avg_latency_s         |          7.611 |      14.374 |
| p95_latency_s         |          8.732 |      15.847 |
| invalid_json_rate     |          0     |       0     |
| avg_context_words     |        460.75  |     857.35  |

## By difficulty

|                          |   em |    f1 |   full_support |
|:-------------------------|-----:|------:|---------------:|
| ('baseline_rag', 'hard') |  0.4 | 0.525 |           0.7  |
| ('graph_rag', 'hard')    |  0.5 | 0.691 |           0.75 |
