from keras.layers import Input, Dense
from keras.models import Model

def main():
	# size of encoded representation
	encoding_dim = 32

	# input placeholder
	input_img = Input(shape=(784,))

	# encoded representation of the input
	layer = Dense(encoding_dim, activation='relu')
	encoded = layer(input_img)

	# decoded reconstruction of the input
	layer = Dense(784, activation='sigmoid')
	decoded = layer(encoded)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)





if __name__ == '__main__':
	main()

