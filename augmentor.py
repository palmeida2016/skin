import os
import imutils
import cv2
import random
import numpy as np
import skimage

class Augmentor():
	def __init__(self, train_path = 'train', test_path = 'test'):
		self.path_to_dirs = [train_path, test_path]
		self.path_to_imgs = [os.path.join(path,folder,img) for path in self.path_to_dirs for folder in os.listdir(path) for img in os.listdir(os.path.join(path,folder))]
		self.img = None
		self.save_path = os.path.dirname(__file__)
		self.width, self.height = 460, 460

	def resize(self):
		self.img = cv2.resize(self.img, (self.width,self.height), interpolation=cv2.INTER_LINEAR)

	def augment(self):
		pass

	def flip(self, axis=[-1,0,1]):
		ax = random.choice(axis)
		self.img = cv2.flip(self.img, ax)

	def rotate(self, angles=[cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]):
		ang = random.choice(angles)
		self.img = cv2.rotate(self.img, ang)

	def cutout(self):
		width = random.randint(50, 100)
		height = random.randint(50, 100)

		x = random.randint(0, self.img.shape[1] - width)
		y = random.randint(0, self.img.shape[0] - height)
		mask = np.ones(self.img.shape)
		mask[y:y+height, x:x+width, :] = 0
		self.img = self.img * mask
		
	def blur(self):
		n = random.randint(1,5)
		self.img = cv2.blur(self.img, (n,n), cv2.BORDER_DEFAULT)

	def noise(self):
		self.img = cv2.GaussianBlur(self.img,(5,5),cv2.BORDER_DEFAULT)

	def saveImage(self, i):
		path = os.path.join(self.save_path, f'test{i}.jpeg')
		cv2.imwrite(path, self.img)

	def test(self):
		path = 'test.jpeg'

		self.img = cv2.imread(path)
		self.resize()
		self.flip()
		self.rotate()
		self.blur()
		self.noise()
		self.cutout()
		self.saveImage(1)







if __name__ == '__main__':
	aug = Augmentor()
	aug.test()
