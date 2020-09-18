# Run ResNet18 model using the Library

### Library Path
[https://github.com/vamsigp/EVA5/tree/master/trainer](https://github.com/vamsigp/EVA5/tree/master/trainer)

### Notebook Path
[https://github.com/vamsigp/EVA5/blob/master/week-8/EVA_AssignmentS8_solution.ipynb](https://github.com/vamsigp/EVA5/blob/master/week-8/EVA_AssignmentS8_solution.ipynb)

#### Final Accuracy - 86.61%

>> EPOCH: 50

>> Loss=0.0020624413155019283 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:45<00:00,  7.44it/s]
/content/EVA5/trainer/test.py:15: UserWarning: This overload of nonzero is deprecated:
	nonzero()
	
>> Consider using one of the following signatures instead:
	nonzero(*, bool as_tuple) (Triggered internally at  /pytorch/torch/csrc/utils/python_arg_parser.cpp:766.)
  misclassified_inds = (is_correct==0).nonzero()[:,0]


>> Accuracy of plane : 88 %

>> Accuracy of   car : 100 %

>> Accuracy of  bird : 68 %

>> Accuracy of   cat : 77 %

>> Accuracy of  deer : 84 %

>> Accuracy of   dog : 77 %

>> Accuracy of  frog : 80 %

>> Accuracy of horse : 91 %

>> Accuracy of  ship : 97 %

>> Accuracy of truck : 97 %

>> Test set: Average loss: 0.0080, Accuracy: 8661/10000 (86.61%)
