from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import utils
import matplotlib.pyplot as plt

import sys
import numpy as np
import pandas as pd


df_gan = pd.read_csv('gan_data.csv')
del(df_gan['Unnamed: 0'])

# Split into train, test and val sets
train, test = train_test_split(df_gan, test_size=0.2)
y_train = train['affiliation']
x_train = train.loc[:, df_gan.columns != 'affiliation']
y_train = y_train.astype(object)
y_test = test['affiliation']
x_test = test.loc[:, df_gan.columns != 'affiliation']

class GAN():

    def __init__(self):
        self.img_rows = len(x_train)
        self.img_cols = len(x_train.columns)
        self.img_shape = (self.img_rows, self.img_cols) # 2-dimension
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_cols), activation='tanh'))#changed
        # model.add(Reshape(self.img_cols)) #deleted

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.img_cols)) #changed
        model.add(Dense(512))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()


        img = Input(shape=self.img_cols) #changed
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=32):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            imgs = x_train.sample(batch_size)

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    def evaluate(self, x_test, y_test):
        score = self.discriminator.evaluate(x_test, y_test)
        print('Test Loss = {}, Test MSE = {}'.format(score[0], score[1]))

    def print_name(self, x_test, y_test):
        y_score = self.discriminator.predict(x_test)
        return y_score

    def precision_recall(self, x_test, y_test):
        from sklearn.metrics import average_precision_score
        from sklearn.metrics import recall_score
        y_score = self.discriminator.predict(x_test)
        y_score = np.where(y_score >= 0.5, 1, 0)
        for i in range(len(y_score)):
            y_score[i].astype(float)
        average_precision = average_precision_score(y_test, y_score)
        average_recall = recall_score(y_test, y_score)
        print('Average precision score: {0:0.2f}'.format(average_precision))
        print('Average recall score: {0:0.2f}'.format(average_recall))
        return average_recall
        # In this specific test case, ROC is meaningless, because no negative y_test

if __name__ == '__main__':

    gan = GAN()
#     res_epoch = []
#     for i in range(0,50):
#         gan.train(epochs=i, batch_size=516)
#         gan.evaluate(x_test, y_test)
#         res_epoch.append(gan.precision_recall(x_test, y_test))
#     plt.plot(res_epoch)
#     plt.show()



    # load in a set of mixed data (both red hatters and unlabled)
    test_df = pd.read_csv('transformed_sample.csv')
    names = test_df['Unnamed: 0']
    del test_df['Unnamed: 0']

    def error_detector(df):
        import numpy as np
        a = []
        b = []
        for i in range(len(df.columns)):
            for j in range(len(df)):
                if type(df.iloc[j, i]) == str:
                    a.append(j)
                    b.append(i)

        for i in a:
            for j in b:
                try:
                    df.iloc[i, j] = int(df.iloc[i, j])
                except:
                    df.iloc[i, j] = 3000  # set a large number as rank for exception

        for ele in df.columns:
            if sum(np.isnan(df[ele])) > 0:
                np.nan_to_num(df[ele], 0)

        return df


    error_detector(test_df)

    from sklearn.metrics import roc_curve, auc
    test_df['affiliation'] = np.where(test_df['affiliation'] >= 1, 1, 0)
    X = test_df.loc[:, test_df.columns != 'affiliation']
    y = test_df['affiliation']
    gan.train(epochs=10, batch_size=512)
    gan.evaluate(X,y)
    gan.precision_recall(X,y)
    print(gan.print_name(X,y))
    print(names)







