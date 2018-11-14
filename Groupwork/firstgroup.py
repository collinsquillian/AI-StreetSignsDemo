import tensorflow as tf

for d in directories:
labels = []
images = []
for f in file_names:
if f.endswith(".ppm")]
def load_directories(data):
directories = [d for d in os.listdir(data)
if os.path.isdir(os.path.join(data, d))]
label_directory = os.path.join(data, d)
file_names = [os.path.join(label_directory, f) 
images.append(skimage.data.imread(f))
return images, labels
for f in os.listdir(label_directory) 
labels.append(int(d))

plt.imshow(images[random_signs[i]])
random_signs=[300, 2250, 3650, 4000]
plt.subplot(1,4,i+1)
for i in range(len(random_signs)):
plt.axis('off')
plt.subplots_adjust(wspace=0.5)