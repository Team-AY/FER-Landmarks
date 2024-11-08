from time import sleep

class DeepLearning_API():
    def eval_video(self, video_path, progress_func, completed_func):
        return self.fake_eval_frame(video_path, progress_func, completed_func)

    def fake_eval_frame(self, video_path, progress_func, completed_func):
        print("Fake Eval Frame")        
        percent = 0
        for i in range(10):
            sleep(0.1)
            percent = percent + 0.1
            progress_func(percent)
        
        completed_func(True)
        return True
        