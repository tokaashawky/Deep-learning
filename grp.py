import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os
from PIL import Image

# Load your image dataset and preprocess as needed
# Assuming 'x_train' contains your image data and 'y_train' contains corresponding labels

path = 'F:/dr_2tlem/deep/brainimage/Testing.csv'
image_data = []
label_data = []
class_labels = {'pituitary': 0, 'meningioma': 1, 'glioma': 2, 'no_tumor': 3}
# Make sure to use a different variable for the loop to avoid overwriting the path
for label in os.listdir(path):
    label_path = os.path.join(path, label)
    for image_filename in os.listdir(label_path):
        image_path = os.path.join(label_path, image_filename)
        # Open the image using PIL
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = image.resize((224, 224))
        # Convert the PIL image to a NumPy array
        image_array = np.array(image)
        # Append the image array and corresponding label to the data lists
        image_data.append(image_array)
        label_data.append(class_labels[label])

path2 = 'brainimage/Testing.csv'
image_data1 = []
label_data1 = []
for label in os.listdir(path2):
    #print(label)  # pituitary
    path1 = os.path.join(path2,label)
    #print(path1)   # /kaggle/input/brain-tumor-mri-dataset/Training/pituitary
    for j in os.listdir(path1):
        #print(j)   # Tr-pi_0532.jpg
        image = Image.open(path1+"/"+j)
        image = image.convert("RGB")
        image = image.resize((224,224))
        image = np.array(image)
        image_data1.append(image)
        label_data1.append(class_labels[label])
        #print(image)        
#-----------------------------------------------------------------------
#shuffle to make the lists inordered to make train on them 
train_paths, train_labels = shuffle(train_paths, train_labels)

#-----------------------------------------------------------------------
# Normalize pixel values to be between 0 and 1 help in classification later
'''
In Python, when working with images represented as NumPy arrays,
the pixel values are often integers ranging from 0 to 255. 
Converting them to floating-point values in the range [0, 1]
is done by dividing by 255.0.
'''
x_train = x_train.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train_one_hot = to_categorical(y_train)

# Define the dimensions of your images
img_shape = x_train.shape[1:]

# Split the dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train_one_hot, test_size=0.2, random_state=42)

# 1. Autoencoder + Dense Layers for Classification
input_img = Input(shape=img_shape)
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# Bottleneck layer
bottleneck = Dense(16, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(bottleneck)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(np.prod(img_shape), activation='sigmoid')(decoded)
decoded = Reshape(img_shape)(decoded)

# Create the autoencoder model
autoencoder1 = Model(input_img, decoded)
autoencoder1.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder1.fit(x_train, x_train, epochs=10, batch_size=128, shuffle=True, validation_data=(x_val, x_val))

# Extract encoder for classification
encoder1 = Model(input_img, bottleneck)

# Create classification model
classification_input1 = encoder1.output
classification_output1 = Dense(4, activation='softmax')(classification_input1)
classification_model1 = Model(encoder1.input, classification_output1)

# Compile the classification model
classification_model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the classification model
classification_model1.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))

# 2. Autoencoder + Dense Layers using Encoder Output for Classification
input_img = Input(shape=img_shape)
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# Bottleneck layer
bottleneck = Dense(16, activation='relu')(encoded)

# Create the autoencoder model
autoencoder2 = Model(input_img, bottleneck)
autoencoder2.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder2.fit(x_train, bottleneck, epochs=10, batch_size=128, shuffle=True, validation_data=(x_val, bottleneck))

# Create classification model using encoder output
classification_input2 = autoencoder2.output
classification_output2 = Dense(4, activation='softmax')(classification_input2)
classification_model2 = Model(autoencoder2.input, classification_output2)

# Compile the classification model
classification_model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the classification model
classification_model2.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))

# 3. Autoencoder + CNN Layers + Dense Layers for Classification
input_img = Input(shape=img_shape)
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# Bottleneck layer
bottleneck = Dense(16, activation='relu')(encoded)

# Decoder
decoded = Dense(32, activation='relu')(bottleneck)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(np.prod(img_shape), activation='sigmoid')(decoded)
decoded = Reshape(img_shape)(decoded)

# Create the autoencoder model
autoencoder3 = Model(input_img, decoded)
autoencoder3.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder3.fit(x_train, x_train, epochs=10, batch_size=128, shuffle=True, validation_data=(x_val, x_val))

# Extract encoder for classification
encoder3 = Model(input_img, bottleneck)

# CNN Layers
cnn_output = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder3.output)
cnn_output = MaxPooling2D((2, 2))(cnn_output)
cnn_output = Flatten()(cnn_output)

# Dense Layers for Classification
classification_input3 = cnn_output
classification_output3 = Dense(4, activation='softmax')(classification_input3)
classification_model3 = Model(encoder3.input, classification_output3)

# Compile the classification model
classification_model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the classification model
classification_model3.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))

# 4. Autoencoder + ResNet50 + CNN Layers + Dense Layers for Classification
input_img = Input(shape=img_shape)
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# Bottleneck layer
bottleneck = Dense(16, activation='relu')(encoded)

# Decoder
decoded = Dense(32, activation='relu')(bottleneck)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(np.prod(img_shape), activation='sigmoid')(decoded)
decoded = Reshape(img_shape)(decoded)

# Create the autoencoder model
autoencoder4 = Model(input_img, decoded)
autoencoder4.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder4.fit(x_train, x_train, epochs=10, batch_size=128, shuffle=True, validation_data=(x_val, x_val))

# Extract encoder for classification
encoder4 = Model(input_img, bottleneck)

# ResNet50 (Transfer Learning)
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=img_shape)
resnet_base.trainable = False

# CNN Layers
cnn_output_resnet = resnet_base(encoder4.output)
cnn_output_resnet = Flatten()(cnn_output_resnet)

# Dense Layers for Classification
classification_input4 = cnn_output_resnet
classification_output4 = Dense(4, activation='softmax')(classification_input4)
classification_model4 = Model(encoder4.input, classification_output4)

# Compile the classification model
classification_model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the classification model
classification_model4.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))

# 5. CNN Model
cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=img_shape))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(4, activation='softmax'))

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))

# 6. ResNet50 (Transfer Learning) along with CNN
resnet_cnn_model = Sequential()
resnet_cnn_model.add(ResNet50(weights='imagenet', include_top=False, input_shape=img_shape))
resnet_cnn_model.add(GlobalAveragePooling2D())
resnet_cnn_model.add(Dense(128, activation='relu'))
resnet_cnn_model.add(Dense(4, activation='softmax'))

# Compile the ResNet50 + CNN model
resnet_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the ResNet50 + CNN model
resnet_cnn_model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))

# 7. Evaluate and Compare Models
eval1 = classification_model1.evaluate(x_val, y_val)
eval2 = classification_model2.evaluate(x_val, y_val)
eval3 = classification_model3.evaluate(x_val, y_val)
eval4 = classification_model4.evaluate(x_val, y_val)
eval_cnn = cnn_model.evaluate(x_val, y_val)
eval_resnet_cnn = resnet_cnn_model.evaluate(x_val, y_val)

# Print evaluation results
print("Evaluation Results:")
print(f"1. Autoencoder + Dense Layers: {eval1}")
print(f"2. Autoencoder + Dense using Encoder Output: {eval2}")
print(f"3. Autoencoder + CNN + Dense Layers: {eval3}")
print(f"4. Autoencoder + ResNet50 + CNN + Dense Layers: {eval4}")
print(f"5
