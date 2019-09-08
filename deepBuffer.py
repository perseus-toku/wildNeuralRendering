from colmap.scripts.python.plyfile import PlyData, PlyElement
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from colmap.scripts.python.read_model import (read_model, qvec2rotmat)


# code to create zbuffer
# assuming folder is the current folder
# need access to colmap, dense and the image dataset with sparse

class ZGenerator(object):
    def __init__(self, dataset, plypath, image_path, sparse_path):
        # read plydata
        self.plydata = PlyData.read(plypath)
        self.cameras, self.images, self.points3D = read_model(sparse_path, ".bin")
        self.image_path = image_path

    def z_buffer(self, idx=1):
        image = self.images[idx]
        # intrinsic camera parameters
        camera = self.cameras[self.images[idx].camera_id]
        w, h = camera.width, camera.height
        intrinsic_matrix = np.zeros((3,3))
        intrinsic_matrix[0,0], intrinsic_matrix[1,1] = camera.params[0], camera.params[1]
        intrinsic_matrix[0,2], intrinsic_matrix[1,2] = camera.params[2], camera.params[3]
        intrinsic_matrix[2,2] = 1.0

        extrinsic_matrix = np.zeros((3,4))
        extrinsic_matrix[:, :3] = qvec2rotmat(image.qvec)
        extrinsic_matrix[:, 3] = image.tvec

        proj_matrix = np.matmul(intrinsic_matrix, extrinsic_matrix)

        rendered = np.zeros((h,w,10)) # For last channel, 0 is depth-buffer, 1-2-3 is R-G-B, 4-5-6 is world coordinate, 7-8-9 is normal vector for each point from dense point cloud
        rendered[:, :, 0] = np.inf

        world_cords = np.zeros((self.plydata.elements[0].count, 4))
        world_cords[:, 0], world_cords[:, 1], world_cords[:, 2], world_cords[:, 3] = self.plydata.elements[0].data['x'], self.plydata.elements[0].data['y'], self.plydata.elements[0].data['z'], 1.0
        projected = np.matmul(proj_matrix, world_cords.transpose()).transpose()
        projected[:, 0] = projected[:, 0]/projected[:, 2]
        projected[:, 1] = projected[:, 1]/projected[:, 2]
        count = 0
        valid_z = 0
        covered = 0
        x_pos, x_neg = 0, 0
        y_pos, y_neg = 0, 0
        z_pos, z_neg = 0, 0
        # print("Total is", self.plydata.elements[0].count)
        for p_i, point in enumerate(self.plydata.elements[0].data):
            x,y,z = int(projected[p_i, 0]), int(projected[p_i, 1]), projected[p_i, 2]
            if x >= 0: x_pos += 1
            if x < 0: x_neg += 1
            if y >= 0: y_pos += 1
            if y < 0: y_neg += 1
            if z < 0: z_neg += 1
            if z >= 0: z_pos += 1
            if x >= 0 and x < w and y >= 0 and y < h:
                count += 1
                if z > 0:
                    valid_z += 1
                    if z < rendered[y,x,0]:
                        if rendered[y,x,0] == np.inf:
                            covered += 1
                        normal_vec = np.array([point[3],point[4],point[5]])
                        rgb_value = np.array([point[6], point[7], point[8]])
                        rendered[y,x,0] = z
                        rendered[y,x,1:4] = rgb_value
                        rendered[y,x,4:7] = np.array([point[0],point[1],point[2]])
                        rendered[y,x,7:] = normal_vec
        return rendered, intrinsic_matrix, extrinsic_matrix, proj_matrix

    def get_deep_buffer(self, idx=1, show_rgb=False):
        rendered, intrinsic_matrix, extrinsic_matrix, proj_matrix = self.z_buffer(idx)
        rendered[rendered[:,:,0]==np.inf] = 0.
        depth = rendered[:,:,0]/np.max(rendered[:,:,0]) * 255
        rgb = rendered[:,:,1:4].astype(np.uint8)
        norm = rendered[:,:,7:] * 255

        # also need original image
        image = self.images[image_idx]
        image_path = os.path.join(self.image_path, image.name)
        original_image = Image.open(image_path)
        # should we concatenate original image to deep buffer or seperately return ?

        deepbuffer = rendered
        deepbuffer[:,:,0] = deepbuffer[:,:,0]/np.max(deepbuffer[:,:,0]) * 255
        # convert rgb color values to int
        deepbuffer[:,:,1:4] = deepbuffer[:,:,1:4].astype(np.uint8)
        deepbuffer = deepbuffer[:,:,7:] * 255

        if show_rgb:
            rgb_img = Image.fromarray(rgb)
            # Turn on interactive mode
            rgb_img.save("show.png")

        return deepbuffer


    def save_deepbuffer_to_disk(self):
        # concatenate original image and deepbuffer and save to disk




def main():
    # need to use absolute path?
    dataset = "south-building"
    plypath = "/Volumes/RundeYang/NeuralRendering/wildNeuralRendering/dense/0/fused.ply"
    sparse_path = "/Volumes/RundeYang/NeuralRendering/wildNeuralRendering/dense/0/sparse/"
    image_path = "/Volumes/RundeYang/NeuralRendering/wildNeuralRendering/dense/0/images/"
    zgenerator = ZGenerator(dataset, plypath, image_path, sparse_path)
    zgenerator.get_deep_buffer(idx=1, show_rgb=True)


if __name__ == "__main__":
    main()
