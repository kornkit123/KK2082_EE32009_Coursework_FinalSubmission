changes made
notice that main misclassification comes from class3 which has the weakest signal

1. made traning for class3 signal x3 for each class 3 signal
    tried with higher value eg. x5 but the result did not get better
2. introduce weighting in training 
    level_weights = torch.tensor([0.05, 0.10, 0.15, 0.20, 0.25, 0.30], device=DEVICE)
3. setting changed due to previous changes
SETTINGS = {2: (0.85, 15), 3: (0.9, 15), 4: (0.9, 25), 5: (0.85, 25), 6: (0.7, 25)}

D2 noise 0.438 | dist 15 | thr 0.7 | F_ident 0.975 F_class 0.955 F_dataset 0.961 | class recall 0.98 0.96 0.95 0.94 0.94
D3 noise 0.648 | dist 15 | thr 0.9 | F_ident 0.972 F_class 0.951 F_dataset 0.958 | class recall 0.98 0.96 0.93 0.94 0.93
D4 noise 1.082 | dist 25 | thr 0.9 | F_ident 0.916 F_class 0.885 F_dataset 0.894 | class recall 0.98 0.96 0.69 0.94 0.93
D5 noise 2.073 | dist 25 | thr 0.8 | F_ident 0.841 F_class 0.733 F_dataset 0.766 | class recall 0.98 0.96 0.14 0.92 0.92
D6 noise 3.065 | dist 25 | thr 0.8 | F_ident 0.800 F_class 0.579 F_dataset 0.645 | class recall 0.93 0.94 0.10 0.70 0.92

Estimated F_Final: 0.804
D1 reference: 2432 spikes
D2: 2753 spikes, class counts [537 567 534 540 575]
D3: 2646 spikes, class counts [528 551 486 521 560]
D4: 3319 spikes, class counts [715 706 412 658 828]
D5: 2284 spikes, class counts [511 573  87 568 545]
D6: 2573 spikes, class counts [646 563 148 627 589]