
import tkinter as tk
import numpy as np

import MLP 

class DrawingApp:
    def __init__(self, master):
        
        self.nn = MLP.MLP(784, 800, 10)
        self.nn.load_net('mnist.ps')

        self.master = master
        master.title("Drawing App")

        self.canvas_width =  280 * 3
        self.canvas_height = 280 * 3
        
        master.resizable(False, False)

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.button_clear = tk.Button(master, text="Clear", command=self.clear_canvas, font=('Arial', 40))
        self.button_clear.pack(side='left')


        self.button_save_array = tk.Button(master, text="Guess", command=self.save_array, font=('Arial', 40))
        self.button_save_array.pack(side='left')

        self.pixels = np.zeros((28, 28), dtype=np.uint8)

    def draw(self, event):
        x, y = event.x, event.y
        r = 20
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        col = x // 30
        row = y // 30
        if col >= 0 and col < 28 and row >= 0 and row < 28:
            self.pixels[row][col] = 1

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pixels = np.zeros((28, 28), dtype=np.uint8)

    def classify_result(self, y):
        return y.flatten().tolist().index(max(y.flatten().tolist()))


    def save_array(self):
        print(f'I think you drew a {self.classify_result(self.nn.feed_forward(np.reshape(self.pixels.flatten(), (-1,1))))}')
        #np.save("drawing.npy", self.pixels) 


root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
