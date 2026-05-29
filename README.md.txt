Order of the notebooks:

1. load_data.ipynb
It includes the loading of data from BOLD5000 dataset (images) and from ds001499 (Open Neuro dataset with the fMRI data)

2. language_models
It uniforms the labels across the three datasets used in the BOLD5000 study (COCO, ImegNet, SUN). 
It saves three dataframes which map the labels of each dataset to the final labels chosen by me.

3. extract_voxels
It analyzes the data from ds001499, which include confounds, rois, anatomical data, and bold signals data.
It applies the GLM to extract the voxels responses from the functional (BOLD) data, which will be used as input for the classification task.

4. mlp
It opens the extracted voxels responses, applies PCA, and gives the PCA-reduced-vectors in input to a MLP for classification. 

5. feature_extraction_resnet_cnn
It implements ResNet-50 and a custom CNN model to extract meaningful features from the visual stimuli (BOLD5000 dataset).
Then it applies Ridge Regressionto find which voxels are most correlated to the images features.

6. brnn
BRNN model which takes in input sequences of voxels, given by masking with couple of ROIs. 
Each node of the sequence has fixed length, given by the top-k voxels, the most correlated to the image features (point 5)

7. multi_branch_gnn
Same as BRNN but with sequences of 5 nodes. Not couple of ROIs anymore (coupled corrispondent ROis of the two hemispheres), but single ROis.
Each branch takes in input a sequence of 5 nodes, one branch -->  one hemisphere.


"utils" includes a file with some functions used in the other notebooks, and some variables kept unchanged in all the notebooks.

"models" includes the path with the weights of the saved trained models.

"data" includes:
    - the BOLD5000 data
    - the .txt files with the best params (each model has undergone a grid search)
    - dataframes of the data (rois, confounds, functionals, ...)
    - voxels responses extracted for each subject, in .pkl format
    - image_features_resnet.npz: images features extracted with ResNet-50
    - all_sessions (.json file): dictionary with the list of all sessions for all the 4 subjects
    - all_stimuli (.txt): list of all the stimuli presented in the right order

"sample_export" includes 5 feature vectors (voxels signals) taken randomly, to be used as example in the Dashboard, 
and the corresponding labels.

All the notebooks are well commented and each step is explained, I hope well.

Also, there is the presentation which better explains the steps of the project.

Thanks :/