# Claim Risk Model Card

## Model Purpose

This model predicts whether a CMS synthetic Medicare claim is likely to be high-cost.

## Dataset

The model uses CMS DE-SynPUF synthetic claims data prepared in Notebook 2.

## Target Variable

The target variable is `high_cost_claim`.

A claim is labeled high-cost if its payment amount is greater than or equal to the 90th percentile threshold created during claims EDA.

## Features Used

The model uses the following non-leakage features:

1. claim_type
2. claim_duration_days
3. has_provider_id
4. has_diagnosis_code

Payment amount fields were excluded from training because the target was created from payment amount.

## Best Model

Best model selected by test ROC-AUC:

Random Forest

## Model Results

                       model  train_accuracy  test_accuracy  train_precision  test_precision  train_recall  test_recall  train_f1  test_f1  train_roc_auc  test_roc_auc  roc_auc_gap   f1_gap
Logistic Regression Baseline        0.926606       0.926105         0.604795        0.602973      0.817572     0.815258  0.695269 0.693227       0.906987      0.905577     0.001410 0.002041
               Random Forest        0.937395       0.937072         0.657711        0.656890      0.810470     0.807116  0.726144 0.724296       0.907089      0.905616     0.001473 0.001848

## Risk and Limitations

1. This model uses synthetic Medicare claims data, not real protected health information.
2. The target is a proxy label based on claim payment amount.
3. Payment amount fields are excluded from training to avoid target leakage.
4. This model should be used for triage and prioritization, not automatic claim decisions.
5. Human review should remain part of any operational workflow.

## Intended Use

This model demonstrates claims risk modeling for benefits administration workflows.
It can support prioritization, analytics, and internal review queues.