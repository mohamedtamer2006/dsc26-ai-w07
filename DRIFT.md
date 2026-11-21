# Drift report

| column | PSI | missing (ref) | missing (current) | drifted |
|---|---|---|---|---|
| `weeks_of_cover` | 1.2173 | 0.0% | 0.0% | True |
| `supplier_lead_time_days` | 3.3172 | 0.0% | 0.0% | True |
| `orders_last_4w` | 0.0037 | 0.0% | 0.0% | False |
| `supplier_fill_rate` | 3.7145 | 3.1% | 15.7% | True |
| `therapeutic_class` | 0.0023 | 0.0% | 0.0% | False |

### Summary for Dr. Mona

Three critical upstream supply-chain features have undergone massive distribution drift—`supplier_fill_rate` (PSI 3.71, with missingness surging from 3.1% to 15.7%), `supplier_lead_time_days` (PSI 3.32), and `weeks_of_cover` (PSI 1.22)—while consumer prescription demand (`orders_last_4w`, PSI 0.004) and therapeutic category mix remained completely stable. This severe drift reflects an external distributor supply shock and rationing across national wholesalers rather than changing patient needs at our 140 pharmacy branches. The surge in missing fill rates indicates that several distributors have abruptly halted fulfillment tracking on constrained chronic medications. Because the machine learning model was calibrated on pre-shock historical supply baselines, serving predictions directly on these shifted inputs would risk severely underestimating shortage risks. The shortage service must immediately route drifted and out-of-range supplier inputs to the deterministic `rule_based_risk` fallback policy, while the AI pod incorporates the post-shock telemetry to retrain and redeploy model version v2.
