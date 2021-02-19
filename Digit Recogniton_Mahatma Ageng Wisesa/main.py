import pygame
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tkinter import *
from tkinter import messagebox

class Pixel:
    def __init__(self, pixels, height, width, background=(255,255,255), click_bg=(0,0,0)):
        self.height = height                        # tinggi window
        self.width = width                          # lebar window
        self.background = background                
        self.click_bg = click_bg 
        self.pixelX, self.pixelY = pixels           # pixel yg dibutuhkan (28x28) atau di konversi   
        self.spanX = self.width // self.pixelX      # panjang setiap pixel (20 pixel monitor)
        self.spanY = self.height // self.pixelY     # panjang setiap pixel (20 pixel monitor)
        self.real_test_set = np.zeros((self.pixelX, self.pixelY)) # tempat test list 28 x28

    def draw_background(self, surface):
        # Buat background selalu putih
        pygame.draw.rect(surface, self.background, (0, 0, self.height, self.width))

    def draw_point(self, surface, x, y):
        # Buat ambil titik dan gambar titik yg dipilih (diklik)
        self.x = x # coordinat x yg diklik
        self.y = y # coordinat y yg diklik
        pygame.draw.rect(
            # gambar titik di 28x28
            surface, self.click_bg, (
                (self.x - self.spanX), # x kiri atas
                (self.y - self.spanY), # y kiri atas
                (self.spanX*2), (self.spanY*2) # dua kali lebih gede
                )
        )

    def convert_to_list(self, coor): 
        # Ubah dari 560x560 ke list 28x28
        # format coor = (x, y); coor[0] -> di X; coor[1] -> di y
        self.row = (coor[1]//self.spanY)    # setiap titik ada 20 pixel untuk 28
        self.col = (coor[0]//self.spanX)    # pembagian integer untuk index di list
        
        # warna hitam masih tumpang tindih
        if self.col <=6: # jika lebih dari 27 integer pass
            #self.real_test_set[self.row+1][self.col+1] = 0.7 # bawah kanan
            pass

        if self.col < 27: # jika kurang dari 27 di X gambar titik kanan
            #self.real_test_set[self.row-1][self.col+1] = 0.7 # atas kanan
            self.real_test_set[self.row][self.col+1] = 0.9  # kanan 240

        if self.row < 27: # jika kurang dari 27 di Y gambar titik bawah
            #self.real_test_set[self.row+1][self.col-1] = 0.7 # bawah kiri
            self.real_test_set[self.row+1][self.col] = 0.9  # bawah 240

        #self.real_test_set[self.row-1][self.col-1] = 0.7 # atas kiri   

        # gambar titik atas kiri tengah         
        self.real_test_set[self.row-1][self.col] = 0.9  # atas 249
        self.real_test_set[self.row][self.col-1] = 0.9  # kiri 240
        self.real_test_set[self.row][self.col] = 1    # tengah
        
    def get_list(self):
        # ambil numpy list
        return self.real_test_set

    def clear_list(self):
        # hapus semua list
        self.real_test_set = np.zeros((self.pixelX, self.pixelY))

    def draw_grid(self, surface):
        # draw grid biar ga bingung 
        for i in range(0, self.width+1, self.spanX):
            # draw y_grid
            pygame.draw.lines(surface, (0,0,0), False, [(i, 0), (i, self.height)], 1)
        for j in range(0, self.height+1, self.spanY):
            # draw x grid
            pygame.draw.lines(surface, (0,0,0), False, [(0, j), (self.width, j)], 1)

    # Masih Rusak
    '''
    def plot_matplotlib(self):
        # plot dalam 28x28
        plt.imshow(self.get_list, cmap="Greys", interpolation="nearest")
        plt.show()
    '''

class Feedback:
    def __init__(self, row, col):
        self.row = row  # np row (jumlah row dalam list)
        self.col = col  # np col (jumah colom dalam list)
        self.new_X_train = np.array([]) # inisiasi np array data X_train baru
        self.new_y_train = np.array([]) # inisiasi np array data Y_train baru
        self.fix_num = 0

    def append(self, new_X, new_y):
        # gabungkan new_X dan new_y untuk training baru ke atribut Feedback
        self.new_X_train = np.append(self.new_X_train, new_X) 
        self.new_y_train = np.append(self.new_y_train, new_y)

    def reshape_X(self):
        # reshape ke (i, j, k) bentuk 3D 
        # (baris per matrix, baris, kolom)
        return self.new_X_train.reshape(len(self.new_y_train), self.row, self.col)

    def get_new_X_train(self):
        # return X_train array
        return self.new_X_train

    def get_new_y_train(self):
        # return y_train array
        return self.new_y_train

    def dataset_to_csv(self):
        # simpan data ke csv data training baru
        np.savetxt(f"new_X_train_{self.fix_num}.csv", self.new_X_train, delimiter=',')
        np.savetxt(f"new_y_train_{self.fix_num}.csv", self.new_y_train, delimiter=',')

    def set_fix_num(self, num):
        self.fix_num = num

    '''
    def show(self, new_X, nrows=1, ncols=3):
        # subplots dalam nrows dan ncols
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        ax = ax.flatten()
        for i in range(len(self.new_y_train)):
            img = new_X[i]
            ax[i].imshow(img, cmap="Greys", interpolation="nearest")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
    '''

def predict_list(pixel_list):
    # load model prediksi
    model = tf.keras.models.load_model('model.model')
    predict = model.predict(
        pixel_list.reshape(1, 28, 28) # rehape jadi format input
        )
    
    print('Probabilitas: ')
    for i in range(len(predict[0])):    # print dari masing-masing nilai probailitasnya
        print(f" {i}: {predict[0][i]}")

    max_prob = np.argmax(predict) # cari index yg isinya maximum
    print(f"Predicted: {max_prob}") # print hasil prediksi

    window = Tk()
    window.withdraw()
    messagebox.showinfo("Prediction", f"Prediction: {max_prob}") 
    window.destroy()

def main():
    run = True

    # TRAINING MODE
    # fixing dataset
    
    #new_y = int(input("What number? "))
    #feedback.set_fix_num(new_y)
    
    while run:

        #print(f"iterations: {iterr}")
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN: # pencet spasi
                #iterr = 0
                px.draw_background(win) # bersehin window sebelumnya
                
                # predicted
                predict_list(px.get_list()) # prediksi dari model
                
                # print(px.get_list())

                # MATPLOTLIB dalam 28x28
                '''
                plt.imshow(px.get_list(), cmap='Greys')
                plt.show()
                '''

                # MODE TRAINING MANUAL
                '''
                manual = str(input("manual train (y/n)? "))
                if manual == 'y':
                    #manual_training = False
                    new_y = int(input("What's the number? "))
                
                    feedback.append(px.get_list(), new_y)
                    print("Appended!")
                '''

                px.clear_list() # hapus list

            if pygame.mouse.get_pressed()[0]: # pencet posisi mouse (klik kiri)
                pos = pygame.mouse.get_pos()
                # print(pos) -> print posisi di x-y pixel monitor
                px.draw_point(win, pos[0], pos[1]) # gambar titik-titk di window
                px.convert_to_list(pos) # ubah ke list 28z8

        pygame.display.update()


pygame.init()
width = height = 560 # pixel monitor sesungguhnya
win = pygame.display.set_mode((width, height)) # set ukuran window

pygame.display.set_caption("Real-time Recognition Test")

px = Pixel((28,28), height, width) # kelas Pixel jadi objek px (inisiasi)
px.draw_background(win) # Buat background jadi putih (untuk kondisi inisial)

# MODE TRAINING MANUAL
feedback = Feedback(28, 28) # objek feedback bentuk 28x28

main() # main loop

# MODE TRANING MANUAL
feedback.dataset_to_csv() # simpan data feedback manual ke csv baru

pygame.quit()
quit()