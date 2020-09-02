Download trained model from
>>ckpt_newModel.h5:[https://drive.google.com/file/d/1-1vnnZz-HSv1sKIz0SQCjULM5hCfuInL/view?usp=sharing]

>>ckpt_FineTune.h5:[https://drive.google.com/file/d/1-4BhkGsm6ttw5y1q0SQGg8WGuGuNPM8w/view?usp=sharing]

# **ckpt_newModel**


Acc for 102 classes of flowers : 0.5831

time for 1 image :7ms





# **ckpt_FineTune**


Acc for 102 classes of flowers : 0.7885

time for 1 image :10ms

# traning 

If you need to re-use the model for other image classification tasks all you need to replace train and test dir paths in following code.
```python

train_ds = train_datagen.flow_from_directory(
    '/content/flower-102diffspecies-dataset/flower_data/train/',

    batch_size=25,
    color_mode="rgb",
    target_size = (150, 150),
    class_mode = 'categorical'
    )
valid_ds = valid_datagen.flow_from_directory(
    '/content/flower-102diffspecies-dataset/flower_data/valid/',
    color_mode="rgb",
    target_size = (150, 150),
    class_mode = 'categorical'
    )
```
and the train and test dir should be arranged as following 
```
train
└── type_A
    └── imz1.jpg
    └── imz2.jpg
    └── imz3.jpg
    └── imz4.jpg
    ...
      
└── type_B
    └── imz1.jpg
    └── imz2.jpg
    └── imz3.jpg
    └── imz4.jpg
    ...
└── type_C
    └── imz1.jpg
    └── imz2.jpg
    └── imz3.jpg
    ...
    
    
valid
└── type_A
    └── imz1.jpg
    └── imz2.jpg
    └── imz3.jpg
    └── imz4.jpg
    ...
      
└── type_B
    └── imz1.jpg
    └── imz2.jpg
    └── imz3.jpg
    └── imz4.jpg
    ...
└── type_C
    └── imz1.jpg
    └── imz2.jpg
    └── imz3.jpg
    ...

```

# Get preduction

```python
img_path = ## path of iamge to preduct 

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image


x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255.0

# Let's run our image through our network
probs = model.predict(x)
```
