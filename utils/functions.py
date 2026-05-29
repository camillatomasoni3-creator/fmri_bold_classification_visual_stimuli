import re, os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split


def show_coco_images(idx, original_image_files, presented_image_files, coco_annotations, orig_path, pres_path, coco_final_labels_df):
  '''
  Show the images: original and presented, and the image with the n bounding boxes.
  '''

  # If the list of files are not sorted, sort them
  original_image_files = sorted(original_image_files)
  presented_image_files = sorted(presented_image_files)

  print(f"Image index {idx} in Original_Images: {original_image_files[idx]}")
  print(f"Image index {idx} in Presented_Images: {presented_image_files[idx]}")

  orig_img_path = os.path.join(orig_path, original_image_files[idx])
  pres_img_path = os.path.join(pres_path, presented_image_files[idx])

  # Find annotations for this image
  # Extract image number from the filename
  img_number_match = re.search(r'(\d+)\.jpg$', original_image_files[idx])
  img_number = int(img_number_match.group(1)) if img_number_match else None

  print(f"Image ID: {img_number}\n")

  boxes = []
  categories = set()

  if img_number in coco_annotations:
      labels = coco_annotations[img_number]
      print(f"Found: {len(labels)} objects in image: {img_number}")

      # Show details of every objects found in the image
      for i, obj in enumerate(labels):
          print(f"\nOggetto {i+1}:")
          cat = obj.get('category_id', 'N/A')
          print(f"  - Category ID: {cat}")
          print(f"  - Area: {obj.get('area', 'N/A')}")
          print(f"  - Bbox: {obj.get('bbox', 'N/A')}")
          # Use the DataFrame to get the category name by category_id
          category_name = coco_final_labels_df.set_index('category_id').loc[cat, 'official_name']
          print(f"  - Class Name: {category_name}")

          categories.add(category_name)

          # extract the boxes
          bbox = obj.get('bbox', [0, 0, 0, 0])
          x, y, w, h = bbox
          boxes.append([x, y, x + w, y + h])

  else:
      print(f"No annotations found for image_id {img_number}")
      print(f"Available image_ids sample: {list(coco_annotations.keys())[:10]}")

  # Show the images: original and presented, and the image with the n bounding boxes
  img_orig = plt.imread(orig_img_path)
  img_pres = plt.imread(pres_img_path)

  print("\n")
  plt.figure(figsize=(11, 6))
  plt.subplot(1, 3, 1)
  plt.imshow(img_orig)
  plt.title("Original Image")
  plt.axis('off')

  plt.subplot(1, 3, 2)
  plt.imshow(img_pres)
  plt.title("Presented Image")
  plt.axis('off')

  plt.subplot(1,3,3)
  plt.imshow(img_orig)
  plt.title("Image with Bounding Boxes")
  if len(boxes)!=0:
    for box in boxes:
      plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none'))
  plt.axis('off')
  
  # Suptitle con categorie uniche
  if categories:
    plt.suptitle("COCO - Categories: " + ", ".join(sorted(categories)), fontsize=12, y=0.83)
  plt.show()


def extract_rows_for_one_subject(subject: int, confounds_df, functionals_df, events_df, rois_df):

    # create boolean mask to extract from each df the rows only referring to one subject
    mask_confounds = (confounds_df['subject'].isin([f'CSI{str(subject)}', f'sub-CSI{str(subject)}']))

    mask_functionals = (functionals_df['subject'].isin([f'CSI{str(subject)}', f'sub-CSI{str(subject)}']))

    mask_events = (events_df['subject'].isin([f'CSI{str(subject)}', f'sub-CSI{str(subject)}']))
    mask_rois = (rois_df['subject'].isin([f'CSI{str(subject)}', f'sub-CSI{str(subject)}']))

    # Apply the masks
    confounds_df_sub = confounds_df[mask_confounds]
    # reset index
    confounds_df_sub = confounds_df_sub.reset_index(drop=True)


    functionals_df_sub = functionals_df[mask_functionals]
    # remove 'localizer' runs from mask_functionals
    mask_localizers_functionals  = (functionals_df_sub['run'].isin(['localizer']))
    functionals_df_sub = functionals_df_sub[~mask_localizers_functionals]
    functionals_df_sub = functionals_df_sub.reset_index(drop=True)

    events_df_sub = events_df[mask_events]
    events_df_sub = events_df_sub.reset_index(drop=True)
    rois_df_sub = rois_df[mask_rois]
    rois_df_sub = rois_df_sub.reset_index(drop=True)

    return confounds_df_sub, functionals_df_sub, events_df_sub, rois_df_sub


def mapping_image_to_label(image_name, df):
    '''
    Args:
        image_name (str): name of an image (the NAME, not its ID/Label int the original dataset)
        df (pd.DataFrame): dataframe mapping the wnid and IDs to the final categories
    Returns the final category associated to that stimulus.
    '''
    # Case 1: find out if the image belongs to ImageNet dataset
    match_imagenet = re.match(r"^(n\d+)(?:_\d+)?", image_name)
    # ^: start of the string
    # n: first letter of the wnid ImageNet
    # \d+: one or more ciphers
    # (n\d+): n+ciphers
    # (?:_\d+)?: other group of _ + ciphers (id of the single image)
    # ? final: makes the last group optional
    
    if match_imagenet:
        imagenet_id = match_imagenet.group(1)
        row = df[df["original_name"] == imagenet_id]
        if not row.empty:
            return row["label"].values[0]
        else:
            print(f"Warning: ImageNet ID not found: {imagenet_id}")
            return None
    
    else:
        # Case 3: everything else (SUN images)
        return "scene"
    


def slice_data(subject: int, subject_data: list, list_roi_slices, target_order):
    '''
    Args:
        subject (int): subject, from 1 to 4.
        subject_data: list of features extracted previously. Length: number of trials extracted.
        list_roi_slices: list of 4 dictionaries.
        target_order: desired order of ROIs to return the data.
    Returns data divided in ROIs.
    '''
    
    assert subject<5 and subject>0

    roi_slices = list_roi_slices[subject-1]

    # Initialize the data to return
    roi_data = {roi: [] for roi in target_order}

    # Loop on the trials
    for trial in subject_data:
        for roi_pair in target_order:
            left_part = trial[roi_slices[roi_pair[0]]]
            right_part = trial[roi_slices[roi_pair[1]]]

            # Concatenate
            roi_concat = np.concatenate([left_part, right_part])

            roi_data[roi_pair].append(roi_concat)

    # Convert in Numpy array
    for roi_pair in roi_data:
        roi_data[roi_pair] = np.stack(roi_data[roi_pair], axis=0)  # (n_trials, n_voxels_left+right)
    
    return roi_data


def slice_data_separated_hemispheres(subject: int, subject_data: list, list_roi_slices, RH_order, LH_order):
    '''
    Args:
        subject (int): subject, from 1 to 4.
        subject_data: list of features extracted previously. Length: number of trials extracted.
        list_roi_slices: list of 4 dictionaries.
        RH_order, LH_order: desired order of ROIs to return the data.
    Returns data divided in ROIs.
    '''
    
    assert subject<5 and subject>0

    roi_slices = list_roi_slices[subject-1]

    # Initialize the data to return
    RH_roi_data = {roi: [] for roi in RH_order}
    LH_roi_data = {roi: [] for roi in LH_order}

    # Loop on the trials
    for trial in subject_data:
        for RH_roi, LH_roi in zip(RH_order, LH_order):
            right_part = trial[roi_slices[RH_roi]]
            left_part = trial[roi_slices[LH_roi]]

            RH_roi_data[RH_roi].append(right_part)
            LH_roi_data[LH_roi].append(left_part)

    
    # Convert in Numpy array
    for roi in RH_roi_data:
        RH_roi_data[roi] = np.stack(RH_roi_data[roi], axis=0)  # (n_trials, n_voxels_right)

    # Convert in Numpy array
    for roi in LH_roi_data:
        LH_roi_data[roi] = np.stack(LH_roi_data[roi], axis=0)  # (n_trials, n_voxels_left)
    
    
    return RH_roi_data, LH_roi_data



def train_val_data_for_regression(subject: int, subjects_data: list, roi, images_features, images_labels, trials_df):
    data_voxels_subject = subjects_data[subject-1]

    # Find the indexes of trials_df corresponding to the subject
    indexes_subject = trials_df.index[trials_df['subject'] == f'CSI{subject}']

    # Slice of the images features
    images_features_subject = images_features[indexes_subject]
    images_labels_subject = images_labels[indexes_subject]

    # take the data for the roi in the arguments
    roi_voxels = data_voxels_subject[roi]

    assert roi_voxels.shape[0] == images_features_subject.shape[0]

    # Divide in train and val
    # X (the input) are the CNN features
    # y (the output to predict) are the voxels responses
    X_train, X_val, Y_train, Y_val, labels_train, labels_val = train_test_split(
        images_features_subject, roi_voxels, images_labels_subject, test_size=0.2, random_state=42
    )
    # print("Train:", X_train.shape, Y_train.shape)
    # print("Val:", X_val.shape, Y_val.shape)

    return X_train, X_val, Y_train, Y_val, labels_train, labels_val



def select_top_voxels(X_train, Y_train, X_val, Y_val, k=100, alpha=10.0):
    '''
    X_train: (n_trials_train, n_features)  -> features CNN
    Y_train: (n_trials_train, n_voxels)   -> voxel responses
    X_val:   (n_trials_val, n_features)
    Y_val:   (n_trials_val, n_voxels)

    k: numero di voxel da selezionare
    alpha: regolarizzazione Ridge
    '''
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_train, Y_train)

    # Predict the entire ROI
    Y_pred = ridge.predict(X_val)

    # Measure the correlation voxel-wise between predictions and data
    scores=[]
    for v in range(Y_train.shape[1]):       # for each voxels
        corr = pearsonr(Y_val[:,v], Y_pred[:,v])[0]
        scores.append(corr)
    
    scores = np.array(scores)
    top_idx = np.argsort(scores)[-k:]
    return top_idx, scores


def select_top_voxels_all_rois(subject: int, roi_order: list, subjects_data: list, k, images_features, images_labels, trials_df):
    '''
    Args:
        subject (int): integer from 1 to 4.
        roi_order: list of tuples (LH_roi, RH_roi)
        subjects_data: list of 4 elements, each containing the voxels data extracted for subject.
        images_features: features extracted from the CNN
        images_labels: images labels
        trials_df (pd.DataFrame): dataframe with all the trials
    Loops over the rois, for each roi tuple performs a multi-output Ridge Regression CV on a subsample of the data, to find the best alpha.
    With the best alpha, performs a voxel-wise regression to find the top-k most representatives voxels.
    Returns the top 100 voxels indexes, and the respective scores.
    '''

    top_idxs_all_rois = []
    scores_all_rois = []
    for roi_pair in roi_order:
        X_train, X_val, Y_train, Y_val, labels_train, labels_val = train_val_data_for_regression(subject=subject, 
                                                                                                 subjects_data=subjects_data, 
                                                                                                 roi=roi_pair, 
                                                                                                 images_features=images_features, 
                                                                                                 images_labels=images_labels, 
                                                                                                 trials_df=trials_df)
        
        # Scale the CNN features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)


        # Grid Search with Ridge Regression on a random subset of data
        # the criterion for chosing the best alpha is r2: maximizing the average R2 on the folds
        # we are looking for the models which better explains the variance in the voxels wrt to the features
        n_samples = 500
        idx = np.random.choice(X_train.shape[0], n_samples, replace=False)
        alphas = [1, 100, 10000, 100000, 1000000]
        ridge_cv = RidgeCV(alphas=alphas, scoring="r2")
        ridge_cv.fit(X_train[idx], Y_train[idx])            # doing Ridge Regression CV on the random subsample
        print(f"Best Alpha for roi:{roi_pair} - subject {subject}:", ridge_cv.alpha_)

        top_idxs, scores = select_top_voxels(X_train, Y_train, X_val, Y_val, k=k, alpha=ridge_cv.alpha_)

        top_idxs_all_rois.append(top_idxs)
        scores_all_rois.append(scores)
    
    return top_idxs_all_rois, scores_all_rois



def extract_final_rois(subject: int, subjects_data: list, roi_order, top_idxs_all_rois_sub, trials_df, images_labels):
    
    data_voxels_subject = subjects_data[subject-1]

    # Find the indexes of trials_df corresponding to the subject
    indexes_subject = trials_df.index[trials_df['subject'] == f'CSI{subject}']
    images_labels_subject = images_labels[indexes_subject]

    final_rois = []

    assert len(top_idxs_all_rois_sub) == 5

    for i,roi_pair in enumerate(roi_order):
        # take the data for the roi in the arguments
        roi_voxels = data_voxels_subject[roi_pair]

        final_roi = roi_voxels[:, top_idxs_all_rois_sub[i]]

        print(f"Shape final ROI {roi_pair} for subject {subject}:", final_roi.shape)

        final_rois.append(final_roi)
    
    return final_rois, images_labels_subject


def extract_final_rois_separated_hemispheres(subject: int, subjects_rh_data: list, subjects_lh_data: list, RH_order, LH_order, 
                                             top_idxs_rh_rois_sub, top_idxs_lh_rois_sub, trials_df, images_labels):
    
    rh_voxels_subject = subjects_rh_data[subject-1]
    lh_voxels_subject = subjects_lh_data[subject-1]

    # Find the indexes of trials_df corresponding to the subject
    indexes_subject = trials_df.index[trials_df['subject'] == f'CSI{subject}']
    images_labels_subject = images_labels[indexes_subject]

    final_rh_rois = []
    final_lh_rois = []

    assert len(top_idxs_rh_rois_sub) == 5 and len(top_idxs_lh_rois_sub) == 5

    for i, (rh_roi, lh_roi) in enumerate(zip(RH_order, LH_order)):
        # take the data for the roi in the arguments
        roi_rh_voxels = rh_voxels_subject[rh_roi]
        roi_lh_voxels = lh_voxels_subject[lh_roi]

        final_rh_roi = roi_rh_voxels[:, top_idxs_rh_rois_sub[i]]
        final_lh_roi = roi_lh_voxels[:, top_idxs_lh_rois_sub[i]]

        print(f"Shape final ROI {rh_roi} for subject {subject}:", final_rh_roi.shape)
        print(f"Shape final ROI {lh_roi} for subject {subject}:", final_lh_roi.shape)

        final_rh_rois.append(final_rh_roi)
        final_lh_rois.append(final_lh_roi)
    
    return final_rh_rois, final_lh_rois, images_labels_subject


def select_top_voxels_rois_separated_hemispheres(subject: int, RH_order: list, LH_order: list, subjects_rh_data: list, subjects_lh_data: list,
                               images_features, images_labels, trials_df, k):
    '''
    Args:
        subject (int): integer from 1 to 4.
        RH_order, LH_order: list of rois (LH_roi, RH_roi)
        subjects_data: list of 4 elements, each containing the voxels data extracted for subject. Both for LH and RH.
        images_features: features extracted from the CNN
        images_labels: images labels
        trials_df (pd.DataFrame): dataframe with all the trials
        k: voxels to extract
    Loops over the rois, for each roi performs a multi-output Ridge Regression CV on a subsample of the data, to find the best alpha.
    With the best alpha, performs a voxel-wise regression to find the top-k most representatives voxels.
    Returns the top 100 voxels indexes, and the respective scores.
    '''

    RH_voxels_subject = subjects_rh_data[subject-1]
    LH_voxels_subject = subjects_lh_data[subject-1]

    # Find the indexes of trials_df corresponding to the subject
    indexes_subject = trials_df.index[trials_df['subject'] == f'CSI{subject}']

    # Slice of the images features
    images_features_subject = images_features[indexes_subject]
    images_labels_subject = images_labels[indexes_subject]




    top_idxs_rh_rois, top_idxs_lh_rois = [], []
    scores_rh_rois, scores_lh_rois = [], []
    for rh_roi, lh_roi in zip(RH_order, LH_order):

        # take the data for the roi in the arguments
        roi_rh_voxels = RH_voxels_subject[rh_roi]
        roi_lh_voxels = LH_voxels_subject[lh_roi]

        assert roi_rh_voxels.shape[0] == images_features_subject.shape[0] and roi_lh_voxels.shape[0] == images_features_subject.shape[0]

        # Divide in train and val
        # X (the input) are the CNN features
        # y (the output to predict) are the voxels responses
        X_train, X_val, Y_train_rh, Y_val_rh, Y_train_lh, Y_val_lh, labels_train, labels_val = train_test_split(
            images_features_subject, roi_rh_voxels, roi_lh_voxels, images_labels_subject, test_size=0.2, random_state=42
        )
        # print("Train:", X_train.shape, Y_train.shape)
        # print("Val:", X_val.shape, Y_val.shape)


        # Scale the CNN features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)


        # Grid Search with Ridge Regression on a random subset of data
        # the criterion for chosing the best alpha is r2: maximizing the average R2 on the folds
        # we are looking for the models which better explains the variance in the voxels wrt to the features
        n_samples = 500
        idx = np.random.choice(X_train.shape[0], n_samples, replace=False)
        alphas = [1, 100, 10000, 100000, 1000000]

        # Regression for RH roi
        ridge_cv = RidgeCV(alphas=alphas, scoring="r2")
        ridge_cv.fit(X_train[idx], Y_train_rh[idx])            # doing Ridge Regression CV on the random subsample
        print(f"Best Alpha for roi:{rh_roi} - subject {subject}:", ridge_cv.alpha_)

        top_idxs_rh, scores_rh = select_top_voxels(X_train, Y_train_rh, X_val, Y_val_rh, k=k, alpha=ridge_cv.alpha_)

        # Regression for LH roi
        ridge_cv = RidgeCV(alphas=alphas, scoring="r2")
        ridge_cv.fit(X_train[idx], Y_train_lh[idx])            # doing Ridge Regression CV on the random subsample
        print(f"Best Alpha for roi:{lh_roi} - subject {subject}:", ridge_cv.alpha_)

        top_idxs_lh, scores_lh = select_top_voxels(X_train, Y_train_lh, X_val, Y_val_lh, k=k, alpha=ridge_cv.alpha_)

        # Append
        top_idxs_rh_rois.append(top_idxs_rh)
        top_idxs_lh_rois.append(top_idxs_lh)
        scores_rh_rois.append(scores_rh)
        scores_lh_rois.append(scores_lh)
    
    return top_idxs_rh_rois, top_idxs_lh_rois, scores_rh_rois, scores_lh_rois



print('Functions imported correctly.')