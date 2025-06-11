# Scripts

- [ros2npy](ros2npy.py): Used to convert ROS1 ZedX bags to numpy arrays to use with the rosnpy dataloader.
- [get_pca_basis](get_pca_basis.py): Used to compute a PCA basis over image folders or any of the datasets. This gives a more robust basis for visualization. If used for other purposes than visualization, then make sure to avoid data contamination and only compute on a disjoint set of data from your testing data.
- [semseg_eval](semseg_eval.py): Script to perform open-vocabulary zero shot semantic segmentation evaluation. Has an option to load external point cloud predictions for evaluation giving the flexibility to evaluate approaches not ported to this repo.
- [srchvol_eval](srchvol_eval.py): Script to perform the search volume online evaluation for the mapping baselines evaluated in RayFronts.
- [summarize_srchvol_eval](summarize_srchvol_eval.py): Rough script used to post process srchvol_eval results and compute AUC metrics and visualizations.