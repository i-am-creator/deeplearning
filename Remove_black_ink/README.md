In this project I used 'flower-102diffspecies-dataset' from kaggel. list there paths seclect 3000 images and move them to a dir.

then build input pipeline and add one of the any masks from masks folder using this function.

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
```

Then i train GAN network to genrate the missing information about the lost pixels 

I fed RGB images with black spots with mask(single channel) (information about the part which needs to be edited)

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
