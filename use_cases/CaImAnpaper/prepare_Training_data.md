**HOW TO GENERATE GROUND TRUTH DATA**
* Step 1
go to script ground_truth_cnmf_seeded.py and generate new ground truth.This will generate a file ending in *match_masks.npz
* Step 2
If you want to match the cnmf-seeded components from  GT with the results of a CNMF run you can use the script match_seeded_gt.py
* Step 3
You might want to clean up the components, you can use the prepare_training_set.py 
* Step 4
In order to train the network use either the train_net_minst.py or train_net_cifar.py


