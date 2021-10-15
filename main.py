import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization,Input
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tqdm import tqdm
import os


class Discriminator(Sequential):
	def __init__(self):
		super().__init__()

		self.num_classes = 23
		self.img_rows, self.img_cols = 48,48
		self.batch_size = 32

		self.epochs = 25
		self.nb_validation_samples = 4002
		self.nb_train_samples = 15557

		self.initArchitecture()
		self.initDataGenerator()
		self.initCallbacks()

	def initArchitecture(self):
		# Block-1

		self.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(self.img_rows,self.img_cols,1)))
		self.add(Activation('elu'))
		self.add(BatchNormalization())
		self.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(self.img_rows,self.img_cols,1)))
		self.add(Activation('elu'))
		self.add(BatchNormalization())
		self.add(MaxPooling2D(pool_size=(2,2)))
		self.add(Dropout(0.2))

		# Block-2 

		self.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
		self.add(Activation('elu'))
		self.add(BatchNormalization())
		self.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
		self.add(Activation('elu'))
		self.add(BatchNormalization())
		self.add(MaxPooling2D(pool_size=(2,2)))
		self.add(Dropout(0.2))

		# Block-3

		self.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
		self.add(Activation('elu'))
		self.add(BatchNormalization())
		self.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
		self.add(Activation('elu'))
		self.add(BatchNormalization())
		self.add(MaxPooling2D(pool_size=(2,2)))
		self.add(Dropout(0.2))

		# Block-4 

		self.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
		self.add(Activation('elu'))
		self.add(BatchNormalization())
		self.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
		self.add(Activation('elu'))
		self.add(BatchNormalization())
		self.add(MaxPooling2D(pool_size=(2,2)))
		self.add(Dropout(0.2))

		# Block-5

		self.add(Flatten())
		self.add(Dense(64,kernel_initializer='he_normal'))
		self.add(Activation('elu'))
		self.add(BatchNormalization())
		self.add(Dropout(0.5))

		# Block-6

		self.add(Dense(64,kernel_initializer='he_normal'))
		self.add(Activation('elu'))
		self.add(BatchNormalization())
		self.add(Dropout(0.5))

		# Block-7

		self.add(Dense(self.num_classes,kernel_initializer='he_normal'))
		self.add(Activation('softmax'))

		print(self.summary())


	def initDataGenerator(self):
		train_data_dir = 'train'
		validation_data_dir = 'test'

		train_datagen = ImageDataGenerator(
							rescale=1./255,
							rotation_range=30,
							shear_range=0.3,
							zoom_range=0.3,
							width_shift_range=0.4,
							height_shift_range=0.4,
							horizontal_flip=True,
							fill_mode='nearest')

		validation_datagen = ImageDataGenerator(rescale=1./255)

		self.train_generator = train_datagen.flow_from_directory(
							train_data_dir,
							color_mode='grayscale',
							target_size=(self.img_rows,self.img_cols),
							batch_size=self.batch_size,
							class_mode='categorical',
							shuffle=True)

		self.validation_generator = validation_datagen.flow_from_directory(
									validation_data_dir,
									color_mode='grayscale',
									target_size=(self.img_rows,self.img_cols),
									batch_size=self.batch_size,
									class_mode='categorical',
									shuffle=True)

	def initCallbacks(self):
		checkpoint = ModelCheckpoint('skin_disease.h5',
									 monitor='val_loss',
									 mode='min',
									 save_best_only=True,
									 verbose=1)

		earlystop = EarlyStopping(monitor='val_loss',
								  min_delta=0,
								  patience=3,
								  verbose=1,
								  restore_best_weights=True)

		reduce_lr = ReduceLROnPlateau(monitor='val_loss',
									  factor=0.2,
									  patience=3,
									  verbose=1,
									  min_delta=0.0001)

		self.callbacks = [earlystop,checkpoint,reduce_lr]


	def train(self):
		self.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])


		self.history = self.fit_generator(
			self.train_generator,
            steps_per_epoch=self.nb_train_samples//self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_data=self.validation_generator,
            validation_steps=self.nb_validation_samples//self.batch_size)


class Generator(Sequential):
	def __init__(self):
		super().__init__()

	def initArchitecture():
		self.add(Dense(units=256, _input_dim=100))
		self.add(LeakyReLU(0.2))


if __name__ == '__main__':
	# gen = Generator()
	dis = Discriminator()
	dis.train()