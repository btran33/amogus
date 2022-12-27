from amongus import Amongus_Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import dlib
import cv2
 

class Affine_Transform():
    def __init__(self, amongus_image:Amongus_Image) -> None:
        self.amongus_image = amongus_image
        self.amongus_features = []
        self.amongus_features_list = []
        self.face_features_list = []

        # setup for getting face features and amongus features
        self.get_amongus_features()
        self.get_all_faces_features()

        self.frames_dir = 'output/frames'

    #=========================== LANDMARK FEATURES SECTION ===========================#
    def get_amongus_features(self):
        with open('data/amongus_points.txt') as f:
            feature_points = f.readlines()

        self.amongus_features = []
        for point in feature_points:
            coord = point.strip()
            x, y = coord.split(',')
            self.amongus_features.append((int(x),int(y)))

        self.amongus_features = np.array(self.amongus_features)
        self.amongus_features -= (175,143) # offset in tool usage (weird)

    def resized_amongus_features(self, image_shape, new_width, new_height):
        height, width, _ = image_shape
        resized_feature_x = self.amongus_features[:,0] * (new_width/width)
        resized_feature_y = self.amongus_features[:,1] * (new_height/height)
        return (resized_feature_x.astype(int), resized_feature_y.astype(int))

    def get_all_faces_features(self):
        self.face_features_list = []
        self.amongus_features_list = []
        bbxs = self.amongus_image.bbxs
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat') 

        for bbx in bbxs:
            a, b, width, height = bbx
            sub_img = self.amongus_image.start_img[b:b+height, a:a+width]

            dets = detector(sub_img, 1) # use detector
            if len(dets) == 0:
                dets = dlib.rectangles()
                dets.append(dlib.rectangle(-32, 0, sub_img.shape[0], sub_img.shape[1]))
            
            features = np.zeros((68, 2), np.int32)
            y, x = sub_img.shape[:2]

            for d in dets:
                shape = predictor(sub_img, d) # get landmark features
                for i in range(68):
                    coord = shape.part(i)
                    # limit coordinate to never go beyond image bound
                    features[i, 0] = min(max(coord.x, 0), x-1)
                    features[i, 1] = min(max(coord.y, 0), y-1)


            # append the face features and resized amongus features
            self.face_features_list.append(features)
            shape = self.amongus_image.amongus_list[0].shape
            self.amongus_features_list.append(np.vstack(self.resized_amongus_features(shape, width, height)).T)

        self.face_features_list = np.array(self.face_features_list)

    def display_features(self, figsize=(17,20)):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(self.amongus_image.start_img)
        axes[0].set_title('Original'), axes[0].set_xticks([]), axes[0].set_yticks([])

        for i, features in enumerate(self.face_features_list):
            a, b, _, _ = self.amongus_image.bbxs[i]
            axes[0].plot(features[:,0] + a, features[:,1] + b, 'o')


        axes[1].imshow(self.amongus_image.amongus_final)
        axes[1].set_title('Amogus Result'), axes[1].set_xticks([]), axes[1].set_yticks([])
        for i, features in enumerate(self.amongus_features_list):
            a, b, _, _ = self.amongus_image.bbxs[i]
            axes[1].plot(features[:,0] + a, features[:,1] + b, 'o')
        plt.show()

    #=========================== AFFINE TRANSFORMATION SECTION ===========================#
    def weighted_avg_pts(self, start_pts, end_pts, percentage):
        return np.array(start_pts * percentage + end_pts * (1 - percentage))

    def weighted_avg_img(self, start_img, end_img, percentage):
        return cv2.addWeighted(start_img, percentage, end_img, 1 - percentage, 0)

    def bilinear_interpolate(self, im, points):
        # derived from https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
        x, y = points
        x = np.asarray(x)
        y = np.asarray(y)

        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, im.shape[1]-1)
        x1 = np.clip(x1, 0, im.shape[1]-1)
        y0 = np.clip(y0, 0, im.shape[0]-1)
        y1 = np.clip(y1, 0, im.shape[0]-1)

        Ia = im[y0, x0]
        Ib = im[y1, x0]
        Ic = im[y0, x1]
        Id = im[y1, x1]

        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        a = np.concatenate([[wa]] * 3, axis=0).T * Ia
        b = np.concatenate([[wb]] * 3, axis=0).T * Ib
        c = np.concatenate([[wc]] * 3, axis=0).T * Ic 
        d = np.concatenate([[wd]] * 3, axis=0).T * Id
        return a + b + c + d

    def region_of_interest(self, points):
        min_x, min_y = int(np.min(points[:,0])), int(np.min(points[:,1]))
        max_x, max_y = int(np.max(points[:,0])) + 1, int(np.max(points[:,1])) + 1
        
        return np.asarray([(x, y) for x in range(min_x, max_x) for y in range(min_y, max_y)])

    def warp(self, start_img, end_img, delaunay:Delaunay, points, triangular_affine_matrix):
        roi = self.region_of_interest(points)
        roi_triangular_idx = delaunay.find_simplex(roi)
        
        for i in range(len(delaunay.simplices)):
            start_pts = roi[i == roi_triangular_idx]
            end_pts = np.dot(triangular_affine_matrix[i],
                            np.vstack((start_pts.T, np.ones(len(start_pts)))))
            x, y = start_pts.T
            # directly warp the input end_img
            end_img[y, x] = self.bilinear_interpolate(start_img, end_pts)

    def tri_affine_matrix(self, start_pts, end_pts, tri_pts):
        for i in tri_pts:
            start_t = np.vstack((start_pts[i, :].T, [1, 1, 1]))
            end_t   = np.vstack((end_pts[i,:].T, [1, 1, 1]))

            mat = np.dot(start_t, np.linalg.pinv(end_t))[:2, :]
            yield mat # stop at each pts

    def warp_image(self, start_img, start_pts, end_pts):
        delaunay = Delaunay(end_pts)
        end_img = np.zeros(start_img.shape, np.uint8)

        matrix = self.tri_affine_matrix(start_pts, end_pts, delaunay.simplices)
        tri_affine = np.asarray(list(matrix))

        self.warp(start_img, end_img, delaunay, end_pts, tri_affine)
        return end_img

    def face_morph_test(self, figsize=(17, 20)):
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(top=0.9, bottom=0, left=0, right=1, wspace=0.01, hspace=0.08)

        start_img = self.amongus_image.start_img
        end_img = self.amongus_image.amongus_final
        bbxs = self.amongus_image.bbxs

        cnt = 1
        for alpha in np.linspace(1, 0, 16):
            # saves time on alpha 0 and 1
            if alpha != 0 and alpha != 1:
                final_assign = start_img.copy()

                # for each bbx, affine transform and blend at given alpha
                for i, bbx in enumerate(bbxs):
                    start_feature_points = self.face_features_list[i]
                    end_feature_points = self.amongus_features_list[i]
                    a, b, width, height = bbx

                    # get the blended points
                    blended_points = self.weighted_avg_pts(start_feature_points, end_feature_points, percentage=alpha)
                    # get the warped results from both end
                    start_face = self.warp_image(start_img[b:b+height, a:a+width], start_feature_points, blended_points)
                    end_face   = self.warp_image(end_img[b:b+height, a:a+width], end_feature_points, blended_points)
                    # alpha blend the results
                    blended_image  = self.weighted_avg_img(start_face, end_face, percentage=alpha)

                    final_assign[b:b+height, a:a+width] = np.where(blended_image != 0, blended_image, start_img[b:b+height, a:a+width])
            else:
                final_assign = (end_img, start_img)[int(alpha)]

            plt.subplot(4,4,cnt)
            plt.imshow(final_assign)
            plt.title(f'alpha = {round(alpha, 4)}', size=20)
            plt.axis('off')
            cnt += 1

        plt.suptitle('Face Morph', size=30)
        plt.show()

    def generate_frames(self, alpha_list):
        frames = [0] * len(alpha_list)
        start_img = self.amongus_image.start_img
        end_img = self.amongus_image.amongus_final
        bbxs = self.amongus_image.bbxs

        for i, alpha in enumerate(alpha_list):
            # saves time on alpha 0 and 1
            if alpha != 0 and alpha != 1:
                final_assign = start_img.copy()

                # for each bbx, affine transform and blend at given alpha
                for j, bbx in enumerate(bbxs):
                    start_feature_points = self.face_features_list[j]
                    end_feature_points = self.amongus_features_list[j]
                    a, b, width, height = bbx

                    # get the blended points
                    blended_points = self.weighted_avg_pts(start_feature_points, end_feature_points, percentage=alpha)
                    # get the warped results from both end
                    start_face = self.warp_image(start_img[b:b+height, a:a+width], start_feature_points, blended_points)
                    end_face   = self.warp_image(end_img[b:b+height, a:a+width], end_feature_points, blended_points)
                    # alpha blend the results
                    blended_image  = self.weighted_avg_img(start_face, end_face, percentage=alpha)

                    final_assign[b:b+height, a:a+width] = np.where(blended_image != 0, blended_image, start_img[b:b+height, a:a+width])
            else:
                final_assign = (end_img, start_img)[int(alpha)]
            frames[i] = final_assign

        if alpha_list[-1] != 0:
            frames.append(end_img)

        return frames