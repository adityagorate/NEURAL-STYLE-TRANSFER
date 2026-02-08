import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Ensure output directory
os.makedirs("output", exist_ok=True)

# -----------------------------
# Image Utilities
# -----------------------------
def load_image(path, max_dim=512):
    img = Image.open(path)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)))
    img = np.array(img)
    img = img[tf.newaxis, :]
    return tf.image.convert_image_dtype(img, tf.float32)

def show_image(img, title=None):
    img = tf.squeeze(img, axis=0)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def save_image(img, path):
    img = tf.squeeze(img, axis=0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)

# -----------------------------
# Load Images
# -----------------------------
content_image = load_image("content2.jpg")
style_image = load_image("style2.jpg")

# -----------------------------
# Model Setup (VGG19)
# -----------------------------
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

def get_model():
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    return tf.keras.Model(vgg.input, outputs)

model = get_model()

# -----------------------------
# Loss Functions
# -----------------------------
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def get_features(image):
    outputs = model(image)
    style_outputs = outputs[:len(style_layers)]
    content_outputs = outputs[len(style_layers):]
    style_features = [gram_matrix(s) for s in style_outputs]
    return style_features, content_outputs

style_targets, _ = get_features(style_image)
_, content_targets = get_features(content_image)

# -----------------------------
# Training Setup
# -----------------------------
image = tf.Variable(content_image)
optimizer = tf.optimizers.Adam(learning_rate=0.02)

style_weight = 1e-2
content_weight = 1e4

@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        style_outputs, content_outputs = get_features(image)

        style_loss = tf.add_n([
            tf.reduce_mean((style_outputs[i] - style_targets[i])**2)
            for i in range(len(style_outputs))
        ])
        style_loss *= style_weight / len(style_layers)

        content_loss = tf.add_n([
            tf.reduce_mean((content_outputs[i] - content_targets[i])**2)
            for i in range(len(content_outputs))
        ])
        content_loss *= content_weight / len(content_layers)

        total_loss = style_loss + content_loss

    grad = tape.gradient(total_loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, 0.0, 1.0))

# -----------------------------
# Training Loop
# -----------------------------
epochs = 10
steps_per_epoch = 100

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        train_step(image)
    print(f"Epoch {epoch+1}/{epochs} completed")

# -----------------------------
# Save Output
# -----------------------------
save_image(image, "output/styled.png")
show_image(image, "Stylized Image")
