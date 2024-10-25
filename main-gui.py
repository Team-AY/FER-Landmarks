import PIL.ImageTk
import customtkinter
import cv2
import PIL
from tkinter import Image

class App(customtkinter.CTk):
    #width = 900*2
    #height = 600
    width=1920
    height=1080

    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue")

    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)

        self.title("Facial Expression Recognition App")
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(False, False)   

        self.main_frame = customtkinter.CTkFrame(self)        

        self.main_frame.grid_columnconfigure(0, weight = 1, pad=0, minsize=self.width/2)
        self.main_frame.grid_columnconfigure(1, weight = 1, pad=0, minsize=self.width/2)
        self.main_frame.grid_rowconfigure(0, weight = 1, pad=0)  

        self.main_frame.grid(row=0, column=0)

        self.controls_frame = customtkinter.CTkFrame(self.main_frame)
        self.controls_frame.grid(row=0, column=0, sticky="ns")             

        self.button1 = customtkinter.CTkButton(self.controls_frame, command=self.on_start)
        self.button1.grid(row=0, column=0)     

        self.camera_frame = customtkinter.CTkFrame(self.main_frame)
        self.camera_frame.grid(row=0, column=1, sticky="ns")        

        self.camera_display = customtkinter.CTkLabel(self.camera_frame, text="")
        self.camera_display.grid(row=0, column=0)        

        self.description_display = customtkinter.CTkLabel(self.camera_frame, text="Hello World")
        self.description_display.grid(row=0, column=0)
        
        self.cap = cv2.VideoCapture(0) 



    def on_start(self):
        self.on_streaming()     

    # code for video streaming
    def on_streaming(self):
        self.camera_display.grid(row=0, column=0)  
        self.description_display.grid_forget()

        ret, img = self.cap.read()
        cv2image= cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(cv2image)
        ImgTks = customtkinter.CTkImage(light_image=img, dark_image=img, size=(self.width/2,self.height)) 
        #ImgTks = PIL.ImageTk.PhotoImage(image=img)
        self.camera_display.imgtk = ImgTks
        self.camera_display.configure(image=ImgTks)
        self.after(20, self.on_streaming)    

if __name__ == "__main__":
    app = App()
    #app.on_streaming()
    app.mainloop()