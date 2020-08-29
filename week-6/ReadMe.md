# Assignment 6

1. with L1 + BN
2. with L2 + BN
3. with L1 and L2 with BN
4. with GBN
5. with L1 and L2 with GBN


some observations - 
1. Switching from L2 to L1 regularization dramatically reduces the delta between test loss and training loss.
2. Switching from L2 to L1 regularization dampens all of the learned weights.
3. Increasing the L1 regularization rate generally dampens the learned weights; however, if the regularization rate goes too high, the model can't converge and losses are very high.
