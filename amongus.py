from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Amongus_Image():
    """
    The Amogus class, generating the amongus image from 
    the original image and perform operations on it
    """
    def __init__(self, start_img, amongus_list, bbxs=None) -> None:
        self.start_img = start_img
        self.bbxs = bbxs
        self.amongus_list = amongus_list

    def generate_amongus(self):
        """
        For every bounding box found, generate a random among us face
        for that box, resized to fit
        """
        if not self.bbxs:
            raise ValueError('Bounding boxes are empty')

        self.amongus_final = self.start_img.copy()
        threads = []

        def threading_amongus(bbx):
            a, b, width, height = bbx
            # resize a randomly chosen amongus face
            amongus_face = cv2.resize(self.amongus_list[np.random.randint(0, len(self.amongus_list))], \
                                      (width, height), interpolation=cv2.INTER_AREA)

            # assign amongus face to the final result
            alpha = amongus_face[:, :, 3] / 255.0
            for channel in range(3):
                blend = (1. - alpha) * self.start_img[b:b+height, a:a+width, channel] + \
                         alpha * amongus_face[:, :, channel]
                self.amongus_final[b:b+height, a:a+width, channel] = blend
        
        for bbx in self.bbxs:
            threads.append(Thread(target=threading_amongus, args=(bbx,)))
            threads[-1].start()
        
        for thread in threads:
            thread.join()
    
    def display_amongus(self, figsize=(17,20)):
        _, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(self.start_img)
        axes[0].set_title('Original'), axes[0].set_xticks([]), axes[0].set_yticks([])
        axes[1].imshow(self.amongus_final)
        axes[1].set_title('Amogus Result'), axes[1].set_xticks([]), axes[1].set_yticks([])
        plt.show()