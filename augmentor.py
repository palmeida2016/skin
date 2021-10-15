import os
import imutils
import cv2

class Augmentor():
	def __init__(self, train_path = 'train', test_path = 'test'):
		self.path_to_dirs = [train_path, test_path]
		self.path_to_imgs = [os.path.join(path,folder,img) for path in self.path_to_dirs for folder in os.listdir(path) for img in os.listdir(os.path.join(path,folder))]

	def resize(self, img):
		pass

	def augment(self):
		pass

	def flip(self, img, axis=[-1,0,1]):
		pass

	def rotate(self, img, angles=[90,180,270]):
		pass

	def cutout(self, img):
		pass

	def centerCrop(self, img):
		pass

	def blur(self, img):
		pass

	def noise(self, img):
		pass


if __name__ == '__main__':
	aug = Augmentor()
