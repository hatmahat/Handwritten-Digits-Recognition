import tensorflow as tf
import numpy as np

class Transfer:
    def __init__(self, X_train, y_train):
        self.X_train = tf.keras.utils.normalize(X_train, axis=1) # normalisasi dari 0-255 jadi 0-1
        self.y_train = y_train
        self.All_X_train = self.X_train     # tempat gabungan semua dataset
        self.All_y_train = self.y_train     # tempat gabungan semua dataset

    def append_new_X(self, new_X_train):
        # append new_X_train ke list semua X -> untuk gabungin dataset lama jadi dataset training baru di All_X_train
        self.All_X_train = np.append(self.All_X_train, new_X_train).reshape(
            (self.All_X_train.shape[0] + new_X_train.shape[0]), 28, 28
        )   # jumlah baris per matrix; baris matrix; kolom matrix

    def append_new_y(self, new_y_train):
        # append new_y ke list semua y -> untuk gabungin dataset lama jadi dataset training baru di All_y_train
        self.All_y_train = np.append(self.All_y_train, new_y_train)

    def get_all_X_train(self):
        # return all X_train
        return self.All_X_train

    def get_all_y_train(self):
        # return all y_train
        return self.All_y_train

    def do_normalize_X(self, norm_X):
        # normalisasi ke 0-1
        return tf.keras.utils.normalize(norm_X, axis=1)

    def do_reshape_X(self, new_X_train, new_y_train):
        # ubah data 1D jadi 3D
        return new_X_train.reshape(len(new_y_train), 28, 28)

def get_csv(new_data_name):
    return np.genfromtxt(new_data_name, delimiter=',')

# load dataset
mnist = tf.keras.datasets.mnist     
(X_train, y_train), (X_test, y_test) = mnist.load_data()

trans = Transfer(X_train, y_train)

# load data training baru
#new_X_train = np.genfromtxt('new_X_train.csv', delimiter=',')
new_X_train = get_csv(r'train_dataset\new_X_train.csv')
new_y_train = np.genfromtxt(r'train_dataset\new_y_train.csv', delimiter=',')
# fixing num ver1 neuron
new_X_train_1 = np.genfromtxt(r'train_dataset\new_X_train_1.csv', delimiter=',')
new_y_train_1 = np.genfromtxt(r'train_dataset\new_y_train_1.csv', delimiter=',')
# fixing num ver2 neuron
new_X_train_2 = np.genfromtxt(r'train_dataset\new_X_train_2.csv', delimiter=',')
new_y_train_2 = np.genfromtxt(r'train_dataset\new_y_train_2.csv', delimiter=',')

# ubah 1D jadi 3D
new_X_train = trans.do_reshape_X(new_X_train, new_y_train)
new_X_train_1 = trans.do_reshape_X(new_X_train_1, new_y_train_1)
new_X_train_2 = trans.do_reshape_X(new_X_train_2, new_y_train_2)
# normalisasi X_test, new_X_train,... (yg belum ke normalisasi langsung di atribut transfer)
X_test = trans.do_normalize_X(X_test)
new_X_train = trans.do_normalize_X(new_X_train)
new_X_train_1 = trans.do_normalize_X(new_X_train_1)
new_X_train_2 = trans.do_normalize_X(new_X_train_2)

# append semua X_train biar jadi 1 list X_test
trans.append_new_X(X_test)
trans.append_new_X(new_X_train)
trans.append_new_X(new_X_train_1)
trans.append_new_X(new_X_train_2)

# append semua y_train biar jadi 1 list y_test
trans.append_new_y(y_test)
trans.append_new_y(new_y_train)
trans.append_new_y(new_y_train_1)
trans.append_new_y(new_y_train_2)

# ambil semua X_train, y_train baru
final_X_train = trans.get_all_X_train()
final_y_train = trans.get_all_y_train()

# neural networks
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # make to one dimension 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(final_X_train, final_y_train, epochs=3)
model.save('model.model')

print(f"Shape new_X_train_1: {new_X_train_1.shape}")
print(f"Shape new_y_train_1: {new_y_train_1.shape}")
print(f"Shape new_X_train_2: {new_X_train_2.shape}")
print(f"Shape new_y_train_2: {new_y_train_2.shape}")

#predict = model.predict(X_test[0].reshape(1, 28, 28))
#print(np.argmax(predict))
#print(X_test.shape)