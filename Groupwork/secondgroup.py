print("*** training images shape ***")
print("*** training images shape ***")
print(images28.shape)
images28=[transform.resize(image,(28,28)) for image in images]
images28=np.array(images28)
print(images28.shape)
images28=[transform.resize(image,(28,28)) for image in images]
images28[random_signs[i]].max()))
for i in range(len(random_signs)):
images28[random_signs[i]].min(), 
print("shape: {0}, min: {1}, max: {2}".format(images28[random_signs[i]].shape, 

plt.subplots_adjust(wspace=0.5)
images28=color.rgb2gray(images28)
plt.axis('off')
plt.subplot(1,4,i+1)
plt.imshow(images28[random_signs[i]],cmap="gray")
for i in range(len(random_signs)):