In this project, I have used a flower-102diffspecies-dataset from Kaggle in which I have randomly selected 3000 images and moved them into a training directory and 250 images from the rest were moved into the test directory.

and add some b/W images to be used as masks 

An input pipeline was build which is used to add one of the many masks into an image from the mask folder.


## Add back spots
```python

def someNoise(imz):
  # imz = np.minimum(imz.numpy(), noiseImg)
  mask = tf.random.shuffle(noise_imzs)[0]
  imz = tf.math.minimum(imz, mask)
  imz = tf.keras.layers.Concatenate(axis=-1)([imz, tf.expand_dims(mask[:, :, 0], -1)])

  return tf.convert_to_tensor(imz)
```


## Input Pipeline

```python
train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)
```

Then, I  had trained the GAN network to generate the missing information about the lost pixels and fed RGB images with mask(single channel) (information about the part which needs to be edited)


## Some output after training

![alt text](https://github.com/i-am-creator/deeplearning/blob/master/Remove_black_ink/imzs/1%20(1).jpeg)
![alt text](https://github.com/i-am-creator/deeplearning/blob/master/Remove_black_ink/imzs/1%20(2).jpeg)
![alt text](https://github.com/i-am-creator/deeplearning/blob/master/Remove_black_ink/imzs/1%20(3).jpeg)
![alt text](https://github.com/i-am-creator/deeplearning/blob/master/Remove_black_ink/imzs/1%20(4).jpeg)
![alt text](https://github.com/i-am-creator/deeplearning/blob/master/Remove_black_ink/imzs/1%20(5).jpeg)
![alt text](https://github.com/i-am-creator/deeplearning/blob/master/Remove_black_ink/imzs/1%20(6).jpeg)
![alt text](https://github.com/i-am-creator/deeplearning/blob/master/Remove_black_ink/imzs/1%20(7).jpeg)
![alt text](https://github.com/i-am-creator/deeplearning/blob/master/Remove_black_ink/imzs/1%20(9).jpeg)
![alt text](https://github.com/i-am-creator/deeplearning/blob/master/Remove_black_ink/imzs/1%20(8).jpeg)


from last example it seems like this model didn't praform well with solid colours becouse it has vary few masks with solid black spots.
if we try with more solid masks then hopefull it will praform batter.

to implement the model any other image you need to create a mask to tell the model which part of the image needs to be edited.
this model is trained on only the images of flowers so maybe it cant perform well on other images. 

## Acknowledgement
- Thanks to the author `[Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros]` who published this great paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [TensorFlow](https://www.tensorflow.org/) which provide many useful tutorials for learning TensorFlow 2.0:
    - [Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/alpha/tutorials/generative/dcgan)
    - [Build a Image Input Pipeline](https://www.tensorflow.org/alpha/tutorials/load_data/images)
    - [Get started with TensorBoard](https://www.tensorflow.org/tensorboard/r2/get_started)
    - [Well documented implementation of The paper](https://www.tensorflow.org/tutorials/generative/pix2pix)
- [Google Colaboratory](https://colab.research.google.com/) which allow us to train the models using free GPUs
- [Kaggel](https://www.kaggle.com/) to providing such an amazing [datasets](https://www.kaggle.com/demonplus/flower-dataset-102) for free 
