
import tkinter as tk
import numpy as np
import os

import MLP 

class DrawingApp:
    def __init__(self, master):
        
        self.nn = MLP.DNN('784x800x10')
        self.nn.load_net('mnist_dnn_50_50_50_10.ps')
        
        self.master = master
        self.master.grid()
        master.title("Drawing App")

        self.canvas_width =  200 * 3
        self.canvas_height = 200 * 3
        
        master.resizable(False, False)

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg='white')
        
        self.canvas.grid(column=0, row=0, sticky=tk.E, columnspan=10, rowspan=10)

        self.canvas.bind("<B1-Motion>", self.draw)

        self.button_clear = tk.Button(master, text="Clear", command=self.clear_canvas, font=('Arial', 40))
        self.button_clear.grid(column=0, row=10, sticky=tk.E)


        self.button_save_array = tk.Button(master, text="Guess", command=self.save_array, font=('Arial', 40))
        self.button_save_array.grid(column=1, row=10, sticky=tk.E)

        self.pixels = np.zeros((28, 28), dtype=np.uint8)
        
        self.table_label = tk.Label(text='   0    1    2    3    4    5    6    7    8    9   ', font=('Arial',40))
        self.table_label.grid(column=13, row = 10, columnspan = 8)
        
        self.table_canvas = tk.Canvas(master, width=975, height=600, bg='white')
        self.table_canvas.grid(column=10, row=0, sticky=tk.W, columnspan=15, rowspan=10)


    def draw(self, event):
        x, y = event.x, event.y
        r = 30 
        
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        col = x // 30
        row = y // 30
        
        #self.canvas.create_rectangle(col * r, row * r, (col+1) * r, (row+1) * r, fill='black')

        if col >= 0 and col < 20 and row >= 0 and row < 20:
            self.pixels[row+4][col+4] = 1

            if (0 < col < 27) and (0 < row < 27):
                if self.pixels[row+1][col] != 1:
                    self.pixels[row+1][col] = 0.5
                if self.pixels[row-1][col] != 1:
                    self.pixels[row-1][col] = 0.5 
                if self.pixels[row][col+1] != 1:
                    self.pixels[row][col+1] = 0.5
                if self.pixels[row][col-1] != 1:
                    self.pixels[row][col-1] = 0.5
                if self.pixels[row+1][col+1] != 1:
                    self.pixels[row+1][col+1] = 0.25
                if self.pixels[row+1][col-1] != 1:
                    self.pixels[row+1][col-1] = 0.25
                if self.pixels[row-1][col+1] != 1:
                    self.pixels[row-1][col+1] = 0.25
                if self.pixels[row-1][col-1] != 1:
                    self.pixels[row-1][col-1] = 0.25
            os.system('clear')
            print(f'I think you drew a {self.classify_result(self.nn.feed_forward(np.reshape(self.pixels.flatten(), (-1,1))))}')
            self.draw_table(self.nn.feed_forward(np.reshape(self.pixels.flatten(), (-1, 1))).flatten().tolist())
    
    def draw_table(self, result):
        self.table_canvas.delete('all')
        for col in range(2*10):
            if (col+1) % 2 == 0:
            
                print(f'Column: {int(round((col+1)/2 - 1))}, Guess: {result[int(round( ( col + 1 ) / 2 - 1 ))]}\n')
                self.table_canvas.create_rectangle(45*col + 14, 600 - result[int(round( ( col + 1 ) / 2 - 1))] * 600, 45*(col+1) + 14, 600, fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pixels = np.zeros((28, 28), dtype=np.uint8)

    def classify_result(self, y):
        return y.flatten().tolist().index(max(y.flatten().tolist()))


    def save_array(self):
        print(f'I think you drew a {self.classify_result(self.nn.feed_forward(np.reshape(self.pixels.flatten(), (-1,1))))}')
        #np.save("drawing.npy", self.pixels) 


root = tk.Tk()
root.geometry('1600x675')
app = DrawingApp(root)
root.mainloop()
