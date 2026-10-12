# End-to-end system evaluation

Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` (base model)
Cases: 91  |  seed 7

## Headline

| metric | value |
|---|---|
| cases | 91 |
| pipeline_route_correct_% | 100.0 |
| false_safe_claims | 0 |
| false_safe_denominator | 54 |
| severity_correct_% | 100.0 |
| disclaimer_% | 100.0 |

## Per category

| category | n | disclaimer_% | drugs_ok_% | grounding_mean | judge_faithful_% | judge_on_topic_% | language_ok_% | latency_p50_s | latency_p95_s | mentions_ok_% | no_dose_number_% | no_false_safe_% | numeric_halluc_% | severity_ok_% | status_ok_% | topic_mentioned_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair_en | 12 | 100.0 | 100.0 | 0.157 | 100.0 | 100.0 | 100.0 | 3.11 | 18.51 | 100.0 |  | 100.0 | 0.0 | 100.0 | 100.0 |  |
| pair_sv | 12 | 100.0 | 100.0 | 0.19 | 100.0 | 100.0 | 100.0 | 4.61 | 5.84 | 100.0 |  | 100.0 | 0.0 | 100.0 | 100.0 |  |
| pair_brand | 4 | 100.0 | 100.0 | 0.19 | 100.0 | 100.0 | 100.0 | 4.07 | 4.55 | 100.0 |  | 100.0 | 0.0 | 100.0 | 100.0 |  |
| adversarial_injection | 8 | 100.0 | 100.0 | 0.187 | 100.0 | 37.5 |  | 3.42 | 4.02 |  |  | 100.0 | 0.0 | 100.0 | 100.0 |  |
| topic_alcohol | 10 | 100.0 | 100.0 | 0.177 | 50.0 | 100.0 | 100.0 | 3.1 | 3.89 | 100.0 |  | 100.0 | 0.0 |  | 100.0 | 100.0 |
| indication | 10 | 100.0 | 100.0 | 0.109 | 90.0 | 80.0 | 100.0 | 3.59 | 5.0 | 100.0 |  |  | 0.0 |  | 100.0 |  |
| side_effects | 6 | 100.0 | 100.0 | 0.24 | 66.7 | 100.0 | 100.0 | 5.23 | 5.87 | 100.0 |  |  | 0.0 |  | 100.0 |  |
| pregnancy_driving | 6 | 100.0 | 100.0 | 0.203 | 66.7 | 100.0 | 100.0 | 3.08 | 3.51 | 100.0 |  |  | 0.0 |  | 100.0 |  |
| refusal_dose | 6 |  |  |  |  |  |  | 0.0 | 0.0 |  | 100.0 |  |  |  | 100.0 |  |
| refusal_selfharm | 4 |  |  |  |  |  |  | 0.0 | 0.0 |  | 100.0 |  |  |  | 100.0 |  |
| unknown_drug | 2 |  |  |  |  |  |  | 0.01 | 0.01 |  |  |  |  |  | 100.0 |  |
| chitchat | 3 |  |  |  |  |  |  | 0.01 | 0.01 |  |  |  |  |  | 100.0 |  |
| followup | 8 | 100.0 | 100.0 | 0.303 | 100.0 | 100.0 | 100.0 | 7.52 | 8.37 |  |  | 100.0 | 0.0 |  | 100.0 | 100.0 |
