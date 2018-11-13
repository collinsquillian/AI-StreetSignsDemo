    # Imports libraries
import os
import matplotlib.pyplot as plt
import random
import skimage
from skimage import data, io, filters, transform, color
import numpy as np
import tensorflow as tf
config=tf.ConfigProto(log_device_placement=True)

    # Load dataset
def load_directories(data):
    directories = [d for d in os.listdir(data) 
                   if os.path.isdir(os.path.join(data, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/StreetSigns"
#training = os.path.join(ROOT_PATH, "Training")
training = "./StreetSigns/Training"
#testing = os.path.join(ROOT_PATH, "Testing")
testing = "./StreetSigns/Testing"

images,labels=load_directories(training)
    #Create histogram of labels
plt.hist(labels, 62)
#plt.show()

random_signs=[300, 2250, 3650, 4000]

for i in range(len(random_signs)):
    plt.subplot(1,4,i+1)
    plt.axis('off')
    plt.imshow(images[random_signs[i]])
    plt.subplots_adjust(wspace=0.5)
#plt.show()
    #print("shape: {0}, min: {1}, max: {2}".format(images[random_signs[i]].shape, 
                                                  #images[random_signs[i]].min(), 
                                                  #images[random_signs[i]].max()))
    #Histogram with individual images
unique_labels=set(labels)
plt.figure(figsize=(15,15))
k = 1
for label in unique_labels:
    image = images[labels.index(label)]
    plt.subplot(8,8,i)
    plt.axis('off')
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    i+=1
    plt.imshow(image)
plt.show()

    #Rescale images in images-array
images28=[transform.resize(image,(28,28)) for image in images]
images28=np.array(images28)
print("*** training images shape ***")
print(images28.shape)
    #Check out resizing
for i in range(len(random_signs)):
    print("shape: {0}, min: {1}, max: {2}".format(images28[random_signs[i]].shape, 
                                                  images28[random_signs[i]].min(), 
                                                  images28[random_signs[i]].max()))

    #Convert images to grayscale
images28=color.rgb2gray(images28)
    #Check out grayscale transformation
for i in range(len(random_signs)):
    plt.subplot(1,4,i+1)
    plt.axis('off')
    plt.imshow(images28[random_signs[i]],cmap="gray")
    plt.subplots_adjust(wspace=0.5)
#plt.show()
    
    # Wok with images
    #print(training)
images=np.array(images)
print(images.ndim)
print(images.size)
print("Itemsize = %s" % (images.itemsize))
print("Totalsize = %s" % (images.nbytes))
print(images[0])

labels=np.array(labels)
print(labels.ndim)
print(labels.size)
print(len(set(labels)))

    #Neural network construction
x = tf.placeholder(dtype=tf.float32, shape=[None,28,28])
x = tf.Print(x,[x])
y = tf.placeholder(dtype=tf.int32, shape=[None])
images_flat=tf.contrib.layers.flatten(x)
logits=tf.contrib.layers.fully_connected(images_flat,62,tf.nn.relu)
loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits))
#loss=tf.Print(loss,[loss],"************* Print loss: ")
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_pred=tf.argmax(logits,1)
print_out = tf.Print(correct_pred, [loss], "Current loss: ")
accuracy=tf.reduce_mean(tf.cast(print_out,tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

    #Plots tf.Graph on port :6006
writer=tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

    #Training the neural network
tf.set_random_seed(1234)
sess=tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

for i in range(1001):
    print('EPOCH', i)
    accuracy_val=sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
    if i % 10==0:
        print("Loss: ", loss)
        print('DONE WITH EPOCH')

    #Evaluating the neural network
sampleIndex=random.sample(range(len(images28)),10)
sampleImage=[images28[i] for i in sampleIndex]
sampleLabel=[labels[i] for i in sampleIndex]

predicted=sess.run([correct_pred], feed_dict={x:sampleImage})[0]
print(sampleLabel)
print(predicted)

fig=plt.figure(figsize=(10,10))
for i in range(len(sampleImage)):
    truth=sampleLabel[i]
    prediction=predicted[i]
    plt.subplot(5,2,i+1)
    plt.axis('off')
    color='green' if truth==prediction else 'red'
    plt.text(40,10,"Truth: {0}\nPrediction: {1}".format(truth, prediction),fontsize=12,color=color)
    plt.imshow(sampleImage[i],cmap="gray")
plt.show()

############################# Testrun against full dataset #############################
test_images,test_labels=load_directories(testing)
test_images28=[transform.resize(image,(28,28)) for image in test_images]
print("*** testing images shape before grayscaling ***")
print(np.array(test_images28).shape)
test_images28=skimage.color.rgb2gray(np.array(test_images28))
print("*** testing images shape ***")
print(test_images28.shape)

predicted=sess.run([correct_pred], feed_dict={x: test_images28})[0]
#print(test_labels)
#print(predicted)
match_count=sum([int(y==y_) for y,y_ in zip(test_labels, predicted)])
totalset=len(test_labels)
ratio=match_count/totalset
print("Total: {0}".format(totalset))
print("Match_count: {0}".format(match_count))
#print("Accuracy: {0}".format(ratio))
#Build accuracy model

sess.close()




    # Intitialize constants
#x1=tf.constant([1,2,3,4])
#x2=tf.constant([5,6,7,8])

    # Multiply constants
#result=tf.multiply(x1,x2)

    # Automatic session handling
#with tf.Session() as sess:
    #output=sess.run(result)
    #print(output)