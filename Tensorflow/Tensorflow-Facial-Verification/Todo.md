This is the stuff to make Facial Verification we need to do
1. Setup                                        
   a. 1.1 install Dependencies
   b. 1.2 import Dependencies
   c. 1.3 create folder structur
2. Collect Positive and Anchors
   a. 2.1 Untar labelled faces in the wild dataset
   b. 2.2 Collect Positive and Anchors Class
3. Load and Preprocess Images
   a. 3.1 get image directories
   b. 3.2 preprocessing Scale and resize images
   c. 3.3 Create Labelled Dataset
4. Model Engineering
   a. 4.1 build func make_embedding
   b. 4.2 Build Distance Layer
   c. 4.3 Make Siamese Model
5. Training
   a. 5.1 Setup Loss and Optimizer
   b. 5.2 Establish Checkpoints
   c. 5.3 Build Train Step Function
   d. 5.4 Build Training Loop
   e. 5.5 Train the Model
6. Evaluate Model
   a. 6.1 Import metrics 
   b. 6.2 Make prdiction
   c. 6.3 Calculation Metrics
7. Save Model
   a. 7.1 Save weights & Reload model
8. Real Time Test
   a. 8.1 Vertification Function
    8.2 OpenCV Real Time Vertification

