if i % 10==0:
writer.add_graph(tf.get_default_graph())
#Training the neural network
tf.set_random_seed(1234)
sampleIndex=random.sample(range(len(images28)),10)
writer=tf.summary.FileWriter('.')
sampleImage=[images28[i] for i in sampleIndex]
sess=tf.InteractiveSession()
for i in range(1001):
print('EPOCH', i)
sess.run(tf.global_variables_initializer())
accuracy_val=sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
if i % 10==0:
print('DONE WITH EPOCH')
#Evaluating the neural network
sampleLabel=[labels[i] for i in sampleIndex]
print("Loss: ", loss)