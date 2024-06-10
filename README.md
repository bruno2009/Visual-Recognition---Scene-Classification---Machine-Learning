# VR . Scene Classification Project . 

## Group
- Bruno Soares de Castro
- Cristiano Augusto Schütz
- Maria Joana Lobo


The Project is structured as:

- Scene_Recognition.ipynb
- Machine_learning - SVM.ipynb 
- Machine_learning - Random-Forest.ipynb



## Scene_Recognition.ipynb

- Dataset loading and splitting
- Feature extraction

This notebook represents the first step, a workflow was defined for extracting SIFT descriptors from images, applying PCA for dimensionality reduction, and using the resulting descriptors for clustering and feature extraction.
	
Feature Extraction:
- Raw SIFT Descriptors: Extracted for various configurations of SIFT parameters (n_hist and n_ori).
```
	n_hist= 2, n_ori = 4   
	n_hist= 4, n_ori = 8   
	n_hist= 6, n_ori = 12    
```
- PCA-transformed SIFT Descriptors: PCA is applied to the raw SIFT descriptors to reduce dimensionality.
- Function BestCluster function was defined to plot the sum of squared distances for different cluster sizes to help identify the optimal number of clusters for KMeans clustering.
```
	Number of clusters: 50 and 150 
```
- Fisher Vectors: Extracted using a GMM trained on the SIFT descriptors.  
	
Histograms are plotted to provide a visual representation of the distribution of visual words.
		
Observations: Used multiprocessing to speed up the extraction process.

#### Generated .npy and .csv

feat_test_50_2_4.npy      
feat_test_50_4_8.npy      
feat_test_50_6_12.npy     
feat_test_150_2_4.npy     
feat_test_150_4_8.npy     
feat_test_150_6_12.npy    

feat_train_50_2_4.npy     
feat_train_50_4_8.npy    
feat_train_50_6_12.npy    
feat_train_150_2_4.npy    
feat_train_150_4_8.npy    
feat_train_150_6_12.npy   

pca_feat_test_50_2_4.npy  
pca_feat_test_50_4_8.npy  
pca_feat_test_50_6_12.npy   
pca_feat_test_150_2_4.npy    
pca_feat_test_150_4_8.npy   
pca_feat_test_150_6_12.npy  

pca_feat_train_50_2_4.npy   
pca_feat_train_50_4_8.npy  
pca_feat_train_50_6_12.npy  
pca_feat_train_150_2_4.npy  
pca_feat_train_150_4_8.npy  
pca_feat_train_150_6_12.npy 


FV_feat_train_50_4_8.npy   
FV_feat_test_50_4_8.npy   
FV_feat_train_150_4_8.npy   
FV_feat_test_150_4_8.npy    

y_train.npy  
y_test.npy

feat_train.csv           
feat_test.csv             


## Machine_learning - SVM.ipynb 


Built several SVM models for different scenarios using dense features, keypoints, PCA, and Fisher Vectors. 

1. **Data Loading and Setup**:
   - Previously exctracted descriptors, PCA descriptors, Fisher vectors, and dense features, are loaded.
   - K-Fold cross-validation and scoring metrics are defined.

2. **SVM Models for Keypoint Features**:
   - Six SVM models are built for different combinations of keypoint features (clusters, n_hist, n_ori).
   - Each model uses a linear SVM classifier and is fine-tuned using GridSearchCV to optimize Cohen's Kappa score.
   - Model performance metrics (Cohen's Kappa, Balanced Accuracy, Accuracy, Classification Report, Confusion Matrix) are computed and displayed for both training and test sets.

3. **SVM Models for PCA-Transformed Keypoint Features**:
   - Similar to the previous step, six SVM models are built using PCA-transformed keypoint features.
   - Model performance metrics are computed and displayed for both training and test sets.

4. **SVM Models for Fisher Vectors**:
   - Two SVM models are built using Fisher vectors.
   - Model performance metrics are computed and displayed for both training and test sets.

5. **SVM Model for Dense Features**:
   - One SVM model is built using only dense features.
   - Model performance metrics are computed and displayed for both training and test sets.

6. **SVM Model for Combined Dense and Keypoint Features**:
   - Six SVM models are built using a combination of dense and keypoint features.
   - Model performance metrics are computed and displayed for both training and test sets.

7. **SVM Model for Combined Dense, Keypoint, and PCA-Transformed Keypoint Features**:
   - Similar to the previous step, six SVM models are built using a combination of dense, keypoint, and PCA-transformed keypoint features.
   - Model performance metrics are computed and displayed for both training and test sets.


## Machine_learning - Random-Forest.ipynb

Same structure but for Random Forest models. 



# Results and Considerations

For better readability the confusion matrixes can be seen in the html's of the ML notebooks and in the Results.pdf which contains a summary of the accuracies obtained for each model.

Altough the overall accuracy of the tested models wasn't very high this project served as a good exercise to understand how different feature extraction techniques and machine learning algorithms perform in scene classification.

The most successful model was "model_svm_dense_key_50_6_12" which means the utilization of 50 clusters, a histogram configuration of 6, and an orientation setting of 12, joined with Dense Features.

The confusion matrixes show that overall the models have an easier time in classifying classes: casino, winecellar and bathroom. This could be attributed to distinctive visual characteristics. 

As future work, there are several avenues to explore for improving model performance. Experimentation with alternative feature extraction methods, model architectures, and data augmentation techniques could lead to better generalization and robustness. Additionally, incorporating domain-specific knowledge or leveraging ensemble learning approaches could enhance classification accuracy.