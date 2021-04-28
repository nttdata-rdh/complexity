# Supplemental Materials for Complexity Estimator

This page provides supplemental materials to a paper, ["Model-based Data-complexity Estimator for Deep Learning Systems"](./aitest.pdf).
We show all results including the ones omitted from the paper due to space limit.

## RQ1. Feature Extraction
### Figure 4
Example of training data with large weight for some common feature obtained in a dense layer (*i.e.*, deep layer.)
![](./figure/Base_maxim_example_actv_6_Horse-1.png)

### Figure 4' (Not shown in the paper)
Example of training data with large weight for ***each*** common feature obtained in a dense layer (*i.e.*, deep layer.)
![](./figure/Base_maxim_example_actv_6_0-1.png)
![](./figure/Base_maxim_example_actv_6_1-1.png)
![](./figure/Base_maxim_example_actv_6_2-1.png)
![](./figure/Base_maxim_example_actv_6_3-1.png)
![](./figure/Base_maxim_example_actv_6_4-1.png)

### Figure 5
Example of training data with large weight for some common feature obtained in a convolution layer (*i.e.*, shallow layer.)
![](./figure/Base_maxim_example_actv_1_rep-1.png)

### Figure 5' (Not shown in the paper)
Example of training data with large weight for ***each*** common feature obtained in a convolution layer (*i.e.*, shallow layer.)
![](./figure/Base_maxim_example_actv_1_0-1.png)
![](./figure/Base_maxim_example_actv_1_1-1.png)
![](./figure/Base_maxim_example_actv_1_2-1.png)
![](./figure/Base_maxim_example_actv_1_3-1.png)
![](./figure/Base_maxim_example_actv_1_4-1.png)


## RQ2. 

### Figure 6
Relationship between prediction accuracy and complexities.
When a model is trained on dataset (a),
as the test data are added in ascending (descending) order of complexities, the prediction accuracy decreases (increases).
![](./figure/Normal_Acc_curve_6-1.png)

### Figure 6' (Not shown in the paper)
Relationship between prediction accuracy and complexities.
We show ***result with a model trained on dataset (b).***
We also show ***results with a model trained on dataset (a) and tested on dataset (c'), (d'), and (e').***
  
Trained on dataset (b)  
![](./figure/Label_lack_Acc_curve_6-1.png)  
Tested on dataset (c')  
![](./figure/Bright_Acc_curve_6-1.png)  
Tested on dataset (d')  
![](./figure/Contrast_Acc_curve_6-1.png)  
Tested on dataset (e')  
![](./figure/Saturation_Acc_curve_6-1.png)

### Figure 7
Proportion of labels in test data with top 10% complexities
![](./figure/High_Recon_Loss_histogram_6-1.png)


### Figure 8
Example of inputs in a training dataset with high complexities.
![](./figure/Example_High_Recon_Train_6-1.png)

### Figure 8' (Not shown in the paper)
Example of inputs ***randomly chosen from a training dataset***.
Images with red frames indicate suspicious data determined by one of authors.
![](./figure/Example_Rand_Recon_Train_annot-1.png)

### Figure 10
Histogram of complexities for various inclusion relations between training and test datasets.
![](./figure/Histogram_dist_cmp_normal_for_paper-1.png)

### Figure 10'
Histogram of complexities for ***all layers and test datasets***.
![](./figure/Histogram_dist_train_on_dataset_a-1.png)

### Figure 10''
Histogram of complexities for ***various models trained on different training datasets***. 
<img src="./figure/Histogram_dist_train_on_dataset_b-1.png" alt="" width="49%"> <img src="./figure/Histogram_dist_brightness-1.png" alt="" width="49%"> <br/>
<img src="./figure/Histogram_dist_contrast-1.png" alt="" width="49%"> <img src="./figure/Histogram_dist_saturation-1.png" alt="" width="49%">
