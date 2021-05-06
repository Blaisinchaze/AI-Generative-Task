import numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def make_generator_model():
    model = tf.keras.Sequential(name='generator')
    # Create the first layer with input 100, output the size of the next layer after reshaping
    model.add(layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # reshape the input size to a different volume, must match the number of weights in the dense layer
    model.add(layers.Reshape((4, 4, 1024)))
    # First Conv2DTranspose Layer, output will be shaped (4*2,4*2,512)
    model.add(layers.Conv2DTranspose(512, (5, 5), (2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # Second Conv2DTranspose Layer, output will be shaped (16,16,256)
    model.add(layers.Conv2DTranspose(256, (5, 5), (2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # Third Conv2DTranspose Layer, output will be shaped (32,32,128)
    model.add(layers.Conv2DTranspose(128, (5, 5), (2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # Final Conv2DTranspose Layer, output will be shaped (64,64,3)
    model.add(layers.Conv2DTranspose(3, (5, 5), (2, 2), padding='same', use_bias=False, activation='tanh'))
    model.summary()
    return model


def make_discriminator_model():
    # Create the actual model as a sequential model
    model = tf.keras.Sequential(name='discriminator')
    # convolve-dropout-activate
    model.add(layers.Conv2D(64, (4, 4), (2, 2), padding='same', use_bias=False, input_shape=(64, 64, 3)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    # convolve-dropout-activate
    model.add(layers.Conv2D(128, (4, 4), (2, 2), padding='same', use_bias=False, input_shape=(64, 64, 3)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    # convolve-dropout-activate
    model.add(layers.Conv2D(256, (4, 4), (2, 2), padding='same', use_bias=False, input_shape=(64, 64, 3)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    # flatten
    model.add(layers.Flatten())
    # output
    model.add(layers.Dense(1))
    model.summary()
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        i = 0
        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        generate_and_save_images(generator, epochs, seed)


def generate_and_save_images(model, epoch, test_input):
    gen_vectors = model(test_input, training=False)

    plt.subplot(4, 4, 1, frameon=False)

    for idx in range(gen_vectors.shape[0]):
        img = numpy.array((gen_vectors[idx] * 127.5) + 127.5)
        img = img.astype('uint8')
        img = Image.fromarray(img)
        plt.subplot(4, 4, idx + 1)
        plt.imshow(img)
        plt.axis('off')

    plt.savefig(os.path.join(os.getcwd(), 'epoch_{:04d}.png'.format(epoch)))
    # plt.show()
    plt.close()


def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


BUFFER_SIZE = 60000
BATCH_SIZE = 64

# Batch and shuffle the data
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(os.getcwd(), "train_images"),
    image_size=(64, 64),
    label_mode=None, shuffle='true'
)

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 500
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

train(train_dataset, EPOCHS)

display_image(EPOCHS)
