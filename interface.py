import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import subprocess

#Main application class
class CroplandCROSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CroplandCROS Noise Cleaner")
        self.geometry("800x600")
        self.input_path = None
        self.output_path = "output.jpg"

        # Create interface elements
        self.create_widgets()

    def create_widgets(self):
        # Input Section
        input_frame = tk.Frame(self)
        input_frame.pack(pady=10)

        tk.Label(input_frame, text="Input Image:").grid(row=0, column=0, padx=5, pady=5)
        browse_btn = tk.Button(input_frame, text="Browse", command=self.browse_image)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=20)

        process_btn = tk.Button(btn_frame, text="Process Image", command=self.process_image, width=20)
        process_btn.grid(row=0, column=0, padx=10, pady=10)

        # Display Section
        self.image_frame = tk.Label(self, text="Image Preview", bg="gray", width=100, height=30)
        self.image_frame.pack(pady=10)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.input_path = file_path
            self.show_image(file_path)

    def process_image(self):
        if not self.input_path:
            messagebox.showerror("Error", "Please select an input image first.")
            return

        # Call the main script
        command = f"python main.py {self.input_path}"
        subprocess.run(command, shell=True, check=True)
        messagebox.showinfo("Success", "Image processed successfully.")
        self.show_image(self.output_path)
    
    def show_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((400, 400), Image.BICUBIC) 
        img_tk = ImageTk.PhotoImage(img)
        self.image_frame.configure(image=img_tk, text="", width=400, height=400)
        self.image_frame.image = img_tk


if __name__ == "__main__":
    app = CroplandCROSApp()
    app.mainloop()
