# Validation report (synthetic data)

## Information Value

dti              1.641
loan_amount      0.449
tenure_months    0.425
income           0.354
delinq_12m       0.032
utilization      0.028
age              0.017
emp_years        0.007

train        AUC=0.824  Gini=0.648  KS=0.505
validation   AUC=0.844  Gini=0.688  KS=0.544
out_of_time  AUC=0.835  Gini=0.670  KS=0.525

## Calibration (out-of-time)

          n  predicted_pd  observed_dr
decile                                
0       497        0.0420       0.0463
1       495        0.0635       0.0768
2       497        0.0866       0.0926
3       495        0.1126       0.1131
4       496        0.1447       0.1835
5       496        0.1928       0.1895
6       497        0.2703       0.3139
7       495        0.3868       0.4162
8       496        0.6780       0.6996
9       496        0.8343       0.8649

## Score PSI (train vs out-of-time): 0.109  (<0.10 stable, 0.10-0.25 monitor, >0.25 shift)

## Feature PSI (train vs out-of-time)

utilization      0.173
delinq_12m       0.116
income           0.083
dti              0.033
loan_amount      0.002
tenure_months    0.002

## Score-band monitoring (SQL)

score_band  accounts  observed_default_pct  predicted_pd_pct
    1_<500      1214                 71.91             69.53
 2_500-549      2222                 22.64             20.31
 3_550-599      1524                  7.22              6.49