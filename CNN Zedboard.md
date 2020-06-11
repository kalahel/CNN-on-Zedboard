# CNN Zedboard

## Architecture 

```python
model = Sequential()
model.add(Conv2D(1, kernel_size=(3, 3), input_shape=input_shape, padding="valid", use_bias=False))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
```

## Inputs

Il est important de diviser par 255 les valeurs des images.

```
28x28 => 26*26 => 13*13 => 64 => 32 => 10
```

Pour l'export des poids il est nécessaire de **transposer** les matrices de poids.

