import PIL.ImageTk
import customtkinter
import cv2
import PIL
from tkinter import Image
from api.landmarks import landmarks

class App(customtkinter.CTk):
    #width = 900*2
    #height = 600
    width=1920
    height=1080
    is_running = False
    landmarks_class = landmarks.Landmarks_API()    
    available_cameras = landmarks_class.list_cameras()
    camera_indexes, camera_names = zip(*available_cameras)

    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue")

    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)

        self.title("Facial Expression Recognition App")
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(False, False)   

        self.main_frame = customtkinter.CTkFrame(self)        

        self.main_frame.grid_columnconfigure(0, weight = 1, pad=0, minsize=self.width/2, uniform='a')
        self.main_frame.grid_columnconfigure(1, weight = 1, pad=0, minsize=self.width/2, uniform='a')
        self.main_frame.grid_rowconfigure(0, weight = 1, pad=0, minsize=self.height, uniform='a')  

        self.main_frame.grid(row=0, column=0, sticky="news")

        self.controls_frame = customtkinter.CTkFrame(self.main_frame)
        self.controls_frame.grid(row=0, column=0, sticky="ns")             

        self.button1 = customtkinter.CTkButton(self.controls_frame, command=self.on_start, text="Start")
        self.button1.grid(row=0, column=0, pady=(10, 10)) 

        self.button2 = customtkinter.CTkButton(self.controls_frame, command=self.on_stop, text="Stop")
        self.button2.grid(row=1, column=0, pady=(10, 10))  
                
        self.dropdown_menu = customtkinter.CTkOptionMenu(self.controls_frame, values=self.camera_names, command=self.on_dropdown_select)        
        self.dropdown_menu.grid(row=2, column=0, pady=(10, 10))  

        self.camera_frame = customtkinter.CTkFrame(self.main_frame)
        self.camera_frame.grid(row=0, column=1, sticky="ns")        

        self.camera_display = customtkinter.CTkLabel(self.camera_frame, text="")
        self.camera_display.grid(row=0, column=0)        

        self.description_display = customtkinter.CTkLabel(self.camera_frame, text="Hello World")
        self.description_display.grid(row=0, column=0)             



    def on_start(self):
        select_camera_name = self.dropdown_menu.get()
        pos = self.camera_names.index(select_camera_name)        
        #self.cap = cv2.VideoCapture(self.camera_indexes[pos])
        self.landmarks_class.open_camera(self.camera_indexes[pos])
        self.is_running = True
        self.on_streaming()     
    
    def on_stop(self):
        self.is_running = False


    # code for video streaming
    def on_streaming(self):
        self.camera_display.grid(row=0, column=0)  
        self.description_display.grid_forget()

        ret, img = self.landmarks_class.get_frame_from_camera()

        if not ret:
            pass # TODO: Handle this

        cv2image = self.landmarks_class.classify_image(img)

        cv2image= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = PIL.Image.fromarray(cv2image)
        ImgTks = customtkinter.CTkImage(light_image=img, dark_image=img, size=(self.width/2,self.height)) 
        #ImgTks = PIL.ImageTk.PhotoImage(image=img)
        self.camera_display.imgtk = ImgTks
        self.camera_display.configure(image=ImgTks)

        if self.is_running:
            self.after(20, self.on_streaming)   
        else:
            self.landmarks_class.close_camera()
            self.camera_display.grid_forget() 
            self.description_display.grid(row=0, column=0)             

    def on_dropdown_select(self, choice):
        print(choice)
             

if __name__ == "__main__":
    app = App()
    #app.on_streaming()
    app.mainloop()