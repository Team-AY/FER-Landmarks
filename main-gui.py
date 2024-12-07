import PIL.Image
import PIL.ImageTk
import customtkinter
import cv2
import PIL
from tkinter import Image
from customtkinter import filedialog
from api.landmarks import landmarks
from api.deeplearning import deeplearning
from api.env_vars import env_loader
from api.email_reports.send_email import send_email_report

from threading import Thread

class App(customtkinter.CTk):
    #width = 900*2
    #height = 600
    width=1920
    height=1080
    is_running = False
    landmarks_class = landmarks.Landmarks_API()    
    deeplearning_class = deeplearning.DeepLearning_API()
    available_cameras = landmarks_class.get_available_cameras()
    availble_models = deeplearning_class.get_available_models()
    user_fullname = None
    user_email = None

    LIVE_DESCRIPTION_TEXT = "This is the Live option for the FER Application.\n" \
                            "You can choose in the Control Panel which camera to use.\n" \
                            "You can choose a real webcam or a virtual webcam.\n\n" \
                            "After that, you can press Start in order to initiate the FER process.\n" \
                            "During the FER process, the application will recognise the expressed feelings in the feed.\n" \
                            "Once you are ready to end the FER process, press the Stop button.\n\n" \
                            "At the end, you will receive a Quick Report of the expressed feelings.\n" \
                            "The Quick Report can be found in the Reports folder.\n"

    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue")

    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)

        self.title("Facial Expression Recognition App")
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(False, False)   

        self.tabview = customtkinter.CTkTabview(self)
        self.tabview.grid(row=0, column=0)
        
        self.tabview.add("Home")
        self.tabview.add("Live") 
        self.tabview.add("Offline") 
        self.tabview.set("Home")  

        # make tabs bigger
        for button in self.tabview._segmented_button._buttons_dict.values():
            button.configure(width=200, height=100, font=('Arial', 30))

        self.on_login()

        self.tab_home_init()
        self.tab_live_init()     
        self.tab_offline_init()      
        
    def tab_home_init(self):
        self.home_main_frame = customtkinter.CTkFrame(self.tabview.tab("Home"))        

        self.home_main_frame.grid_columnconfigure(0, weight = 1, pad=0, minsize=self.width, uniform='a')        
        self.home_main_frame.grid_rowconfigure((0, 1, 2), weight = 1, pad=0)          

        self.home_title_display = customtkinter.CTkLabel(self.home_main_frame, text="Facial Expression Recognition App", font=('Arial', 52))
        self.home_title_display.grid(row=0, column=0, pady=(10, 10))

        self.home_image_display = customtkinter.CTkLabel(self.home_main_frame, text="")
        self.home_image_display.grid(row=1, column=0, pady=(10, 10))

        img = PIL.Image.open("gui/images/home_picture.png")
        ImgTks = customtkinter.CTkImage(light_image=img, dark_image=img, size=(self.width/4,self.height/2)) 
        #ImgTks = PIL.ImageTk.PhotoImage(image=img)
        self.home_image_display.imgtk = ImgTks
        self.home_image_display.configure(image=ImgTks)

        self.home_description_display = customtkinter.CTkLabel(self.home_main_frame, 
                                                               text=f"Hello {self.user_fullname} ({self.user_email})\nThis app lets you recognize expressed feelings in two domains:\n1. Live - RealTime Webcam Feed\n2. Offline - Uploaded Files",
                                                               font=('Arial', 28))
        self.home_description_display.grid(row=2, column=0, pady=(10, 10))

        self.home_credits_display = customtkinter.CTkLabel(self.home_main_frame, 
                                                           text="Research & Development By:\n Almog Rabani\n Yakir Hasid",
                                                           font=('Arial', 24))
        self.home_credits_display.grid(row=3, column=0, pady=(10, 10))

        self.home_main_frame.grid(row=0, column=0)  

    def tab_live_init(self):
        self.rt_main_frame = customtkinter.CTkFrame(self.tabview.tab("Live"))        

        # split screen to 2 columns
        self.rt_main_frame.grid_columnconfigure(0, weight = 1, pad=0, minsize=self.width/3, uniform='a')
        self.rt_main_frame.grid_columnconfigure(1, weight = 1, pad=0, minsize=self.width/3, uniform='a')
        self.rt_main_frame.grid_columnconfigure(2, weight = 1, pad=0, minsize=self.width/3, uniform='a')
        self.rt_main_frame.grid_rowconfigure(0, weight = 1, pad=0, minsize=self.height)  

        self.rt_main_frame.grid(row=0, column=0, sticky="news")

        self.controls_frame = customtkinter.CTkFrame(self.rt_main_frame)
        self.controls_frame.grid(row=0, column=0, sticky="n")             

        self.dropdown_menu = customtkinter.CTkOptionMenu(self.controls_frame, values=self.available_cameras)        
        self.dropdown_menu.grid(row=0, column=0, padx=(10, 10), pady=(10, 10), columnspan=2) 

        self.button_start = customtkinter.CTkButton(self.controls_frame, command=self.on_start, text="Start")
        self.button_start.grid(row=1, column=0, padx=(10, 10), pady=(10, 10))
        self.button_start.configure(width=100, height=50)

        self.button_stop = customtkinter.CTkButton(self.controls_frame, command=self.on_stop, text="Stop")
        self.button_stop.grid(row=1, column=1, padx=(10, 10), pady=(10, 10))         
        self.button_stop.configure(width=100, height=50)

        self.slider_description = customtkinter.CTkLabel(self.controls_frame, text="Sample Rate", font=('Arial', 28))
        self.slider_description.grid(row=2, column=0, padx=(10, 10), pady=(10, 10), columnspan=2)         
        self.slider_description.configure(width=100, height=50)

        self.slider = customtkinter.CTkSlider(self.controls_frame, from_=1, to=10, orientation="horizontal", command=self.on_slide_change)
        self.slider.grid(row=3, column=0, padx=(10, 10), pady=(10, 10), columnspan=2)         
        #self.slider.configure(width=100, height=50)   
        self.slider.set(1)
        
        self.slider_value = customtkinter.CTkLabel(self.controls_frame, text="1", font=('Arial', 28))
        self.slider_value.grid(row=4, column=0, padx=(10, 10), pady=(10, 10), columnspan=2)         
        self.slider_value.configure(width=100, height=50)                        

        self.camera_frame = customtkinter.CTkFrame(self.rt_main_frame)
        self.camera_frame.grid(row=0, column=1, sticky="n", columnspan=2)        

        self.camera_display = customtkinter.CTkLabel(self.camera_frame, text="")
        self.camera_display.grid(row=0, column=0)        

        self.description_display = customtkinter.CTkLabel(self.camera_frame, text=self.LIVE_DESCRIPTION_TEXT, font=('Arial', 28), justify="left")
        self.description_display.grid(row=0, column=0)

    def tab_offline_init(self):
        self.offline_main_frame = customtkinter.CTkFrame(self.tabview.tab("Offline"))        

        # split screen to 2 columns
        self.offline_main_frame.grid_columnconfigure(0, weight = 1, pad=0, minsize=self.width/3, uniform='a')
        self.offline_main_frame.grid_columnconfigure(1, weight = 1, pad=0, minsize=self.width/3, uniform='a')
        self.offline_main_frame.grid_columnconfigure(2, weight = 1, pad=0, minsize=self.width/3, uniform='a')
        self.offline_main_frame.grid_rowconfigure(0, weight = 1, pad=0, minsize=self.height)  

        self.offline_main_frame.grid(row=0, column=0, sticky="news")

        self.offline_controls_frame = customtkinter.CTkFrame(self.offline_main_frame)
        self.offline_controls_frame.grid(row=0, column=0, sticky="n")   
        
        self.models_menu = customtkinter.CTkOptionMenu(self.offline_controls_frame, values=self.availble_models)        
        self.models_menu.grid(row=0, column=0, padx=(10, 10), pady=(10, 10), columnspan=2)             

        self.button_load = customtkinter.CTkButton(self.offline_controls_frame, command=self.on_load, text="Load Video")
        self.button_load.grid(row=1, column=0, padx=(10, 10), pady=(10, 10), columnspan=2)
        self.button_load.configure(width=100, height=50)        

        self.progres_label = customtkinter.CTkLabel(self.offline_controls_frame, text="Processing Progress:", font=('Arial', 28), justify="left")
        self.progres_label.grid(row=2, column=0, padx=(10, 10), pady=(10, 10))
        self.progres_label.configure(width=100, height=50) 

        self.progressbar = customtkinter.CTkProgressBar(self.offline_controls_frame)
        self.progressbar.set(0)
        self.progressbar.grid(row=2, column=1, padx=(10, 10), pady=(10, 10))

        self.completed_label = customtkinter.CTkLabel(self.offline_controls_frame, text="Completed", font=('Arial', 28), justify="left")
        self.completed_label.grid(row=3, column=0, padx=(10, 10), pady=(10, 10), columnspan=2)
        self.completed_label.configure(width=100, height=50)
        self.completed_label.grid_forget()

        self.process_frame = customtkinter.CTkFrame(self.offline_main_frame)
        self.process_frame.grid(row=0, column=1, sticky="n", columnspan=2)                 

    def on_slide_change(self, value):
        self.slider_value.configure(text=int(value))

    def on_start(self):
        self.landmarks_class.sample_rate = int(self.slider.get())
        select_camera_name = self.dropdown_menu.get()
        pos = self.available_cameras.index(select_camera_name)        
        self.landmarks_class.open_camera(pos)
        self.is_running = True
        self.on_streaming()     
    
    def on_stop(self):
        self.is_running = False
        filename, current_datetime, most_common_emotion = self.landmarks_class.quick_report(['bar', 'time'])
        send_email_report(filename=filename, current_datetime=current_datetime, most_common_emotion=most_common_emotion, user_fullname=self.user_fullname, user_email=self.user_email)

    def on_load(self):
        filename = filedialog.askopenfilename() 

        # user didn't select a video
        if filename == "":
            return

        model_name = self.models_menu.get()   
        print(filename)

        th = Thread(target=self.deeplearning_class.eval_video, args=(filename, model_name, self.user_fullname, self.user_email, self.progressbar.set, self.display_completed_label))

        th.start()

    def display_completed_label(self, is_completed):
        if is_completed:
            self.completed_label.grid(row=3, column=0, padx=(10, 10), pady=(10, 10), columnspan=2)
        else:
            self.completed_label.grid_forget()
            
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

    def on_login(self):
        self.user_fullname = customtkinter.CTkInputDialog(text="Please enter your full name:", title="Login").get_input()
        self.user_email = customtkinter.CTkInputDialog(text="Please enter your email address:", title="Login").get_input()  

# POSTER V2 Requirement
class RecorderMeter1(object):
    pass

# POSTER V2 Requirement
class RecorderMeter(object):
    pass       
             

if __name__ == "__main__":
    app = App()
    #app.on_streaming()
    app.mainloop()