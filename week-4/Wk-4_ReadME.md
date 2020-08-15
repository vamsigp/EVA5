<b>Strategy </b>

Have tried with Multiple notebooks.

1. EVA5_Session_4.ipynb

	a. Max Accuracy = 99.46%
	
	b. Max Accuracy at Epoch = 17
	
	c. Total number of Parameters = 14308
	
	d. Final Accuracy  = 99.40%
	
2. EVA5_Session_4_1.ipynb

	a. Max Accuracy = 99.42%
	
	b. Max Accuracy at Epoch = 17 and 20
	
	c. Total number of Parameters = 10538
	
	d. Final Accuracy  = 99.42%
	
3. EVA5_Session_4_3.ipynb

	a. Max Accuracy = 90.4%
	
	b. Max Accuracy at Epoch = 19
	
	c. Total number of Parameters = 11242 
	
	d. Final Accuracy  = 99.34%
	
<b>Notes</b>
EVA5_Session_4 and EVA5_Session_4_1 follows the same strategy with varying channel numbers as output.

		[CONVOLUTION] -> [ReLU] -> [BatchNormalization] -> [Dropout] ->
		[CONVOLUTION] -> [ReLU] -> [BatchNormalization] -> [Dropout] ->
		[1D CONVOLUTION] -> [Maxpooling] ->
		[CONVOLUTION] -> [ReLU] -> [BatchNormalization] -> [Dropout] ->
		[CONVOLUTION] -> [ReLU] -> [BatchNormalization] -> [Dropout] ->
		[1D CONVOLUTION] -> [Maxpooling] ->
		[CONVOLUTION] -> [GAP] -> [OUTPUT]
				

EVA5_Session_4_3 follows the same strategy but without 1D CONVOLUTION, but during training the accuracy is not consistent, It achieves 99.4% but not consistent. so I would like not to submit this notebook for evaluation.

For the both the models Accuracy hists 99.4% for the intermediate epoch.

We can consider intermediate epoch model by means of callback and earlystopping. As they were not yet taught i didnt used them.

-----------------------------------------------------

<b>Log from the Model Training Data - </b>

<b><i>EVA5_Session_4.ipynb<i><b>

 0%|          | 0/938 [00:00<?, ?it/s]
Current Epoch - 1

/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:48: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.047176141291856766 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.26it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0464, Accuracy: 9846/10000 (98.46%)

Current Epoch - 2
loss=0.0780990794301033 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.27it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0345, Accuracy: 9878/10000 (98.78%)

Current Epoch - 3
loss=0.002460039220750332 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 56.95it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0271, Accuracy: 9908/10000 (99.08%)

Current Epoch - 4
loss=0.024753522127866745 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 56.75it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0229, Accuracy: 9923/10000 (99.23%)

Current Epoch - 5
loss=0.01676030084490776 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 56.92it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0221, Accuracy: 9932/10000 (99.32%)

Current Epoch - 6
loss=0.04855436459183693 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.41it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0224, Accuracy: 9917/10000 (99.17%)

Current Epoch - 7
loss=0.03951498866081238 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 56.56it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0209, Accuracy: 9925/10000 (99.25%)

Current Epoch - 8
loss=0.0044266460463404655 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.66it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0197, Accuracy: 9938/10000 (99.38%)

Current Epoch - 9
loss=0.02735779993236065 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.50it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0229, Accuracy: 9930/10000 (99.30%)

Current Epoch - 10
loss=0.008900566957890987 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.62it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0205, Accuracy: 9934/10000 (99.34%)

Current Epoch - 11
loss=0.005095400381833315 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 56.85it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0240, Accuracy: 9926/10000 (99.26%)

Current Epoch - 12
loss=0.0015881878789514303 batch_id=937: 100%|██████████| 938/938 [00:17<00:00, 55.00it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0189, Accuracy: 9940/10000 (99.40%)

Current Epoch - 13
loss=0.02101808972656727 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.07it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0199, Accuracy: 9937/10000 (99.37%)

Current Epoch - 14
loss=0.4437441825866699 batch_id=937: 100%|██████████| 938/938 [00:17<00:00, 53.94it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0197, Accuracy: 9937/10000 (99.37%)

Current Epoch - 15
loss=0.00038264362956397235 batch_id=937: 100%|██████████| 938/938 [00:17<00:00, 53.93it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0204, Accuracy: 9940/10000 (99.40%)

Current Epoch - 16
loss=0.058442845940589905 batch_id=937: 100%|██████████| 938/938 [00:17<00:00, 54.35it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0210, Accuracy: 9928/10000 (99.28%)

Current Epoch - 17
loss=0.04892563447356224 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 55.56it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0186, Accuracy: 9946/10000 (99.46%)

Current Epoch - 18
loss=0.002372716087847948 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 56.15it/s]
  0%|          | 0/938 [00:00<?, ?it/s]
Test set: Average loss: 0.0180, Accuracy: 9941/10000 (99.41%)

Current Epoch - 19
loss=0.0004614020581357181 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 55.74it/s]
Test set: Average loss: 0.0186, Accuracy: 9940/10000 (99.40%)


-----------------------------------------------------
-----------------------------------------------------

<b><i>EVA5_Session_4_1.ipynb<i><b>



  0%|          | 0/938 [00:00<?, ?it/s]

Current Epoch - 1

/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:48: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.05725257098674774 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.49it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0501, Accuracy: 9826/10000 (98.26%)

Current Epoch - 2

loss=0.0057900347746908665 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.63it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0335, Accuracy: 9891/10000 (98.91%)

Current Epoch - 3

loss=0.10417187958955765 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.80it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0335, Accuracy: 9891/10000 (98.91%)

Current Epoch - 4

loss=0.013756644912064075 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.82it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0268, Accuracy: 9919/10000 (99.19%)

Current Epoch - 5

loss=0.03951449319720268 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.76it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0266, Accuracy: 9925/10000 (99.25%)

Current Epoch - 6

loss=0.004069901071488857 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.69it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0221, Accuracy: 9930/10000 (99.30%)

Current Epoch - 7

loss=0.047970570623874664 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.71it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0232, Accuracy: 9931/10000 (99.31%)

Current Epoch - 8

loss=0.014847401529550552 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.65it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0238, Accuracy: 9925/10000 (99.25%)

Current Epoch - 9

loss=0.04967889562249184 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.57it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0233, Accuracy: 9921/10000 (99.21%)

Current Epoch - 10

loss=0.0101939020678401 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.58it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0202, Accuracy: 9939/10000 (99.39%)

Current Epoch - 11

loss=0.11924013495445251 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.71it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0215, Accuracy: 9931/10000 (99.31%)

Current Epoch - 12

loss=0.10965795069932938 batch_id=937: 100%|██████████| 938/938 [00:28<00:00, 32.73it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0202, Accuracy: 9938/10000 (99.38%)

Current Epoch - 13

loss=0.007552015129476786 batch_id=937: 100%|██████████| 938/938 [00:29<00:00, 31.60it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0209, Accuracy: 9935/10000 (99.35%)

Current Epoch - 14

loss=0.003199747297912836 batch_id=937: 100%|██████████| 938/938 [00:29<00:00, 31.47it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0202, Accuracy: 9937/10000 (99.37%)

Current Epoch - 15

loss=0.006409753579646349 batch_id=937: 100%|██████████| 938/938 [00:29<00:00, 31.47it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0194, Accuracy: 9941/10000 (99.41%)

Current Epoch - 16

loss=0.0007668439648114145 batch_id=937: 100%|██████████| 938/938 [00:30<00:00, 31.07it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0191, Accuracy: 9940/10000 (99.40%)

Current Epoch - 17

loss=0.004384682513773441 batch_id=937: 100%|██████████| 938/938 [00:30<00:00, 30.94it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0195, Accuracy: 9942/10000 (99.42%)

Current Epoch - 18

loss=0.005486841779202223 batch_id=937: 100%|██████████| 938/938 [00:30<00:00, 30.97it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0197, Accuracy: 9941/10000 (99.41%)

Current Epoch - 19

loss=0.031584613025188446 batch_id=937: 100%|██████████| 938/938 [00:30<00:00, 31.03it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0204, Accuracy: 9939/10000 (99.39%)

Current Epoch - 20

loss=0.0022444722708314657 batch_id=937: 100%|██████████| 938/938 [00:30<00:00, 31.11it/s]

Test set: Average loss: 0.0195, Accuracy: 9942/10000 (99.42%)



-----------------------------------------------------
-----------------------------------------------------


<b><i>EVA5_Session_4_3.ipynb<i><b>




  0%|          | 0/938 [00:00<?, ?it/s]

Current Epoch - 1

/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:43: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.030695807188749313 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 56.67it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0418, Accuracy: 9860/10000 (98.60%)

Current Epoch - 2

loss=0.03154710307717323 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.15it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0328, Accuracy: 9898/10000 (98.98%)

Current Epoch - 3

loss=0.00909727718681097 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.33it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0321, Accuracy: 9902/10000 (99.02%)

Current Epoch - 4

loss=0.11574703454971313 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.84it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0272, Accuracy: 9920/10000 (99.20%)

Current Epoch - 5

loss=0.004912194330245256 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 58.56it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0266, Accuracy: 9920/10000 (99.20%)

Current Epoch - 6

loss=0.024216104298830032 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 58.73it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0260, Accuracy: 9915/10000 (99.15%)

Current Epoch - 7

loss=0.1216861680150032 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 59.12it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0242, Accuracy: 9926/10000 (99.26%)

Current Epoch - 8

loss=0.0014931863406673074 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.56it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0239, Accuracy: 9921/10000 (99.21%)

Current Epoch - 9

loss=0.015587904490530491 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 58.77it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0252, Accuracy: 9913/10000 (99.13%)

Current Epoch - 10

loss=0.003984502982348204 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 59.03it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0235, Accuracy: 9928/10000 (99.28%)

Current Epoch - 11

loss=0.00043390935752540827 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 58.41it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0230, Accuracy: 9925/10000 (99.25%)

Current Epoch - 12

loss=0.040385063737630844 batch_id=937: 100%|██████████| 938/938 [00:15<00:00, 59.10it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0229, Accuracy: 9929/10000 (99.29%)

Current Epoch - 13

loss=0.0036745171528309584 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 58.21it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0191, Accuracy: 9938/10000 (99.38%)

Current Epoch - 14

loss=0.0030221296474337578 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 58.35it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0205, Accuracy: 9935/10000 (99.35%)

Current Epoch - 15

loss=0.000444365490693599 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.86it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0220, Accuracy: 9933/10000 (99.33%)

Current Epoch - 16

loss=0.02879936806857586 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 58.30it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0221, Accuracy: 9925/10000 (99.25%)

Current Epoch - 17

loss=0.004171364009380341 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.99it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0226, Accuracy: 9926/10000 (99.26%)

Current Epoch - 18

loss=0.010186350904405117 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.73it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0225, Accuracy: 9934/10000 (99.34%)

Current Epoch - 19

loss=0.0015729654114693403 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 57.55it/s]
  0%|          | 0/938 [00:00<?, ?it/s]

Test set: Average loss: 0.0203, Accuracy: 9940/10000 (99.40%)

Current Epoch - 20

loss=0.18221880495548248 batch_id=937: 100%|██████████| 938/938 [00:16<00:00, 58.16it/s]

Test set: Average loss: 0.0214, Accuracy: 9934/10000 (99.34%)

