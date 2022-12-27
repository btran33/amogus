from affine_transform import Affine_Transform
import moviepy.editor as mpe
import cv2
import matplotlib.pyplot as plt

# referencing this code https://stackoverflow.com/questions/43048725/python-creating-video-from-images-using-opencv
class Amongus_Video():
    def __init__(self, affine_transform:Affine_Transform, 
                       alpha_smooth:(lambda x: (12**x-1)/11)) -> None:
        self.affine_transform = affine_transform
        self.alpha_smooth = alpha_smooth

        self.audio = mpe.AudioFileClip('data/sus_sfx.mp3')

    def generate_video(self):
        num_frames = round(self.audio.duration * 10)
        alpha_list = [self.alpha_smooth(alpha/num_frames) for alpha in reversed(range(num_frames + 1))]

        frames = self.affine_transform.generate_frames(alpha_list)

        # save the first frame as shape reference
        plt.imsave(f'./output/frames/{0}.jpg', frames[0])
        h, w, layer = cv2.imread('./output/frames/0.jpg').shape
        out = cv2.VideoWriter('output/output.avi', 0, 10.0, (w, h))

        # output avi
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out.release()

        # overlap audio
        video = mpe.VideoFileClip('output/output.avi')
        audio = mpe.AudioFileClip('data/sus_sfx.mp3')
        final = video.set_audio(audio)
        final.write_videofile('output/final_output.mp4')