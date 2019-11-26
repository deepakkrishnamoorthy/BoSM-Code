# BoSM-Code
Readme for execution - Bag-of-Sequential-Models Based Embeddings for video anomaly detection

# Step 1: (Add the folder RealtimeHofHogReleas to matlab directory before executing this step to avoid unncessary errors on video codec and neccesary feature extraction functions)
MBH Feature Extraction:
1.	Open cvsegmentsmbh.m in matlab editor.
2.	Change the directory of training/videos (comment provided in the code)
3.	Save the final data in the form of segments (ex:25 frames per segment)
# Step 2:
Training the HMM models (Bag of Sequential Models)
# Download HMM tool from the link provided - (https://github.com/gabrielhuang/murphy-hmm) and add the folders to matlab working directory

1.	Open cvhmmconv.m file to train the conventional HMM model and the classification code is also available in this particular matlab file.
2.	To train the model based on the proposed idea, kindly open cvproposed.m. 
3.	Set the number of models to be trained in the second **for** loop statement.
4.	For ex- if there are totally 600 training segments and you need to train about 60 sequential models with this training data, then the value of "i" in the first for loop will be 60 (for i = 1:60) and value of j will be 10 (for j = 1:10). 
5.	The above step will 10 segments per batch from the whole training/testing data to built the model. Based on this, about 60 submodels will be trained. The command to normalize the data is also provided in the code.

6.	Once this step is completed, change the numCls (number of classes) variable to 60, since 60 sequential models has to be built to form the final vectorial representation, which conveys that the dimension of training/testing matrices will be 60.
7.	Once the data is ready (check featMain variable for correctness), add the downlaoded HMM tool folder under MATLAB Directory and run the for loop to train the models and obtain the attributes and save them. Link for HMM tool - https://github.com/gabrielhuang/murphy-hmm

# Step 3: 

Forming the testing and training embeddings

1.	After saving the required attributes of modelling (priorC, transmatC, muC, SigmaC, mixmatC), load the training and testing data and execute the loop in sections %%%Forming training Embeddings and %%%Forming testing Embeddings.
2.	The number of samples count should be provided in "i", and the number of models count in "j" (Note: the j value should be same as the dimension of model attributes mentioned in the above step.

# Step 4: 
Classification

1.	After forming the testing and training embeddings, a SVM model can be built in any of the Desired platform to obtain the results.


