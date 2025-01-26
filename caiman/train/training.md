HOW TO GENERATE GROUND TRUTH DATA TO TRAIN THE NETWORK

Step 1: Go to ground_truth_cnmf_seeded.py and generate new ground truth. This generates a file ending in match_masks.npz
Step 2: Go to match_seeded_gt.py IF you want to match the cnmf-seeded components from GT with the results of a CNMF run
Step 3: Go to prepare_training_set.py IF you might want to clean up the components
Step 4: Train the network from  train_cnn_model_pytorch.ipynb (train_cnn_model_keras.ipynb not in use)