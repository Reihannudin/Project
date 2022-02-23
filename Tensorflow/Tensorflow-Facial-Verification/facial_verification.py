'''# 1 Setup'''

# 1.1 install Dependencies
# pip install tensorflow
# pip install open cv
# pip install matplotlib

# 1.2 import Dependencies
# import standard dependencies
from dis import dis
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow Dependencies - Functional API
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1.3 create folder structur
# setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
# make directories
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)



'''# 2 Collect Positive and Anchors'''

# 2.1 Untar labelled faces in the wild dataset
# Labelled Faces in the Wild: http://vis-www.cs.umass.edu/lfw/

# uncompress Tar GZ labelled Face in the wild dataset
# !tar -xf lfw.tgz
'''
#  move LFW Images to the following repository data/negative
# for directory in os.listdir('lfw'):
#     for file in os.listdir(os.path.join('lfw',directory)):
#         EX_PATH = os.path.join('lfw', directory, file)
#         NEW_PATH = os.path.join(NEG_PATH, file)
#         os.replace(EX_PATH,NEW_PATH)
#         
# for directory in os.listdir('lfw'):
#     for file in os.listdir(os.path.join('lfw',directory)):
#         print(os.path.join('lfw',directory,file))
#         print(os.path.join(NEG_PATH, file))
# 
# we are done do it this
'''

# 2.2 Collect Positive and Anchors Class
# Import uuid library to generate uniqe images names
import uuid

# Establish a connection to the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened:
    ret, frame = cap.read()
    
    # cut down frame to 250 x 250
    frame = frame[120:120+250,200:200+250,:]
    
    # collecting anchors
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # create the uniqe file path
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # write out anchor image 
        cv2.imwrite(imgname, frame)
        
    # collecting positive
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # create the uniqe file path
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        # write out positive image 
        cv2.imwrite(imgname, frame)
        
    # show image back to screen
    cv2.imshow("Image Collection", frame)
    
    # Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    
# release the webcam
cap.release()
# close the image show frame
cv2.destroyAllWindows()



'''# 3 Load and Preprocess Images'''

# 3.1 get image directories
anchor = tf.data.Dataset.list_files(ANC_PATH+'\*.jpg').take(255)
positive = tf.data.Dataset.list_files(POS_PATH+'\*.jpg').take(255)
negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(255)

# Testing
# dir_test = anchor.as_numpy_iterator()
# print(dir_test.next())

# 3.2 preprocessing Scale and resize images
def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img,(100,100))
    # scale image to be between 0 and 1
    img = img/255.0
    # return image
    return img

# Testing
# img = preprocess('data\\anchor\\263aaefc-9254-11ec-b94e-d4258bca19e8.jpg')
# img.numpy().max()
# 
# plt.imshow(img)

# 3.3 Create Labelled Dataset
# example
# (anchor, positive) => 1,1,1,1,1
# (anchor, negative) => 0,0,0,0,0

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))

# Testing
data = positives.concatenate(negatives)
# sample = data.as_numpy_iterator()
# exampple = sample.next()
# exampple

# 3.4 Build Train and Test Partitions
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

# Testing
# res = preprocess_twin(*exampple)
# plt.imshow(res[1])
# res[2]

# build dataloader pipline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

# Training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)



'''# 4. Model Engineering'''

# 4.1 Build Embedding Layer
# build func make_embedding
def make_embedding():
    # input
    inp = Input(shape=(100,100,3), name='input_image')
    
    # first block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64,(2,2), padding='same')(c2)
    
    # Third block
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64,(2,2), padding='same')(c3)
    
    # final embedding block
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return Model(inputs=[inp], outputs=[d1], name='embedding')

# Testing embedding
embedding = make_embedding()
# embedding.summary()

# 4.2 Build Distance Layer
# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init Methode - inheritance
    def __init__(self, **kwargs):
        super().__init__()
        
    # Magic happens here similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# l1 = L1Dist()
# l1(anchor_embedding, validation_embedding)


# 4.3 Make Siamese Model
# build func make_siamese_model
def make_siamese_model():
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # vaidation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distance = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # classification layer
    classifier = Dense(1, activation='sigmoid')(distance)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
    
# Testing siamese_mode
siamese_model = make_siamese_model()
# siamese_model.summary()



'''# 5. Training'''

# 5.1 Setup Loss and Optimizer
# setup loss
binary_cross_loss = tf.losses.BinaryCrossentropy(from_logits=True)
# setup optimizers
opt = tf.keras.optimizers.Adam(1e-4) #0,0001

# 5.2 Establish Checkpoints
checkpoints_dir = './training_checkpoints'
checkpoints_prefix = os.path.join(checkpoints_dir, 'ckpt')
checkpoints = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# 5.3 Build Train Step Function
@tf.function 
def train_step(batch):
    
    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get another and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # forward pass 
        y_pred = siamese_model(X, training = True)
        # calculate loss
        loss = binary_cross_loss(y, y_pred)
    print(loss)
    
    # calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # calculate up date weights and aply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    # return loss
    return loss

# 5.4 Build Training Loop
def train(data, EPOCHS):
    # loop thourgh epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

    # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            train_step(batch)
            progbar.update(idx+1)
        
    # save checkpoints 
    if epoch % 10 == 0:
        checkpoints.save(file_prefix=checkpoints_prefix)
        

# 5.5 Train the Model
EPOCHS = 50
# train(train_data, EPOCHS)



'''# 6. Evaluate Model'''

# 6.1 Import metrics 
# Import metric calculations
from tensorflow.keras.metrics import Precision, Recall

# 6.2 Make prdiction
# Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

# Make Prediction
y_pred =siamese_model.predict([test_input, test_val])

# Post processing the result
# with Pythonic syntax
[1 if prediction > 0.5 else 0 for prediction in y_pred]

# Basic way
#result = []
#for prediction in y_pred:
#    if prediction > 0.5:
#        result.append(1)
#    else:
#        result.append(0)
        
        
# 6.3 Calculation Metrics
# Creating a metric object

# with Recall
n = Recall()

# with Precision
# n = Precision()

# Calculation the recall value
n.update_state(y_true, y_pred)
print(n.result().numpy())


# Visualizing
# Set plot size 
plt.figure(figsize=(10,8))

# Set first subplot
plt.subplot(1,2,1)
plt.imshow(test_input[0])

# Set second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[0])

# Renders cleanly
# plt.show()

# Testing
# # value input images
# print("\nInput images")
# # validation data
# print("\nValidation data")
# print(test_val)
# # label
# print("\nLabel ")
# print(y_true)
# 
# print("\nPrediction ")
# print(result)
# 
# print(y_pred)
# 
print(n.result().numpy())


'''# 7. Save Model'''
# Save weights
siamese_model.save('siamesemodelv1.h5')

# Reload model 
# siamese_model = tf.keras.models.load_model('siamesemodelv2.h5', 
#                                      custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# Make predictions with reloaded model
siamese_model.predict([test_input, test_val])

# View model summary
# siamese_model.summary()

'''# 8. Real Time Test '''
# 8.1 Vertification Function
def verify(model, detection_threshold, verification_threshold):
    # Build result array
    results = []
    for image in os.listdir(os.path.join('application_data', 'vertification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'vertification_images', image))
        
        # Make predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
        
    # Detection Threshold: Metrics above which a prediction is considered positive samples
    detection = np.sum(np.array(results) >  detection_threshold)
    
    # Vertification Threshold: Proportion of positive predictions / total positive samples
    verification = detection/ len(os.listdir(os.path.join('application_data','vertification_images')))
    verified = verification > verification_threshold
    
    return results, verified


# 8.2 OpenCV Real Time Vertification
# activate webcam
cap = cv2.VideoCapture(0)
while cap.isOpened:
    ret , frame = cap.read()
    frame = frame[120:120+250, 200:200+250, :]
    
    cv2.imshow('Verification', frame)
    
    # Verification Trigger
    if cv2.waitKey(10) & 0XFF == ord('v'):
        # save input image to application_data/input_image folder
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results , verified = verify(siamese_model, 0.5, 0.5)
        print(verified)
    
    # Breaking gracefully
    elif cv2.waitKey(10) & 0XFF == ord('q'):
        break
  
# release the webcam
cap.release()
# close the image show frame
cv2.destroyAllWindows()

print(np.squeeze(results) > 0.5)

print(np.sum(np.squeeze(results) > 0.5))