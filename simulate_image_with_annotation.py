import numpy as np
from matplotlib import path
from scipy.ndimage import gaussian_filter
from PIL import Image
import math
import os
import shutil
from tqdm import tqdm
from utils import get_coord, get_random_points, get_bezier_curve, rtnorm
from mycoco_json_utils import create_info, create_license, create_images, create_categories, get_segmentation, get_mask_annotation

def fiber(image_id, nfiber, coord, center_arr, img, sigma, Nx, Ny): # tag changed to image_id
    global sub_mask_details
    global annotation_id
    alpha = np.zeros((nfiber, 1))
    L = np.zeros((nfiber, 1))
    W = np.zeros((nfiber,1))
    Density = np.zeros((nfiber,1))
    
    for i in range(nfiber):
    	alpha[i] = np.random.uniform(-math.pi/2, math.pi/2)
    	L[i] = np.random.normal(40, 5)
    	W[i] = 2*2
    	Density[i] = rtnorm(0,1,0.92,0.3)
    
    all_fiber_mask = np.zeros((Ny, Nx))
    
    for i in range(nfiber):
        p1 = np.vstack((-W[i]/2,-L[i]/2))
        p2 = np.vstack((W[i]/2,-L[i]/2))
        p3 = np.vstack((W[i]/2,L[i]/2))
        p4 = np.vstack((-W[i]/2,L[i]/2))
        
        theta = alpha[i]
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        center = np.expand_dims(center_arr[i], axis=1)
        q1 = np.dot(R,p1) + center
        q2 = np.dot(R,p2) + center
        q3 = np.dot(R,p3) + center
        q4 = np.dot(R,p4) + center
        
        V_xy = np.hstack((q1,q2,q3,q4))
        p = path.Path(np.transpose(V_xy))
        fiber_pixl = p.contains_points(coord)
        fiber_pixl = np.reshape(fiber_pixl, (Ny, Nx)) - 0.0
        fiber = fiber_pixl*Density[i]
         
        img = np.maximum(img, fiber)
        all_fiber_mask = np.maximum(all_fiber_mask, fiber_pixl)
        
        sub_mask_details['image_id'].append(image_id+1)
        sub_mask_details['category_id'].append(1)
        sub_mask_details['pixl_arr'].append(fiber_pixl)
        sub_mask_details['id'].append(annotation_id)
        annotation_id += 1

    return img, all_fiber_mask

def particles(image_id, npart, coord, center_arr, img, sigma, Nx, Ny, rad=0.3, edgy=0):
    all_part_mask = np.zeros((Ny, Nx))
    global sub_mask_details
    global annotation_id
    for j in range(npart):
        center = center_arr[j]
        a = get_random_points(n=4, scale=30) + center
        s, c = get_bezier_curve(a, rad=rad, edgy=edgy)
        Density = rtnorm(0,1,0.92,0.3)
        
        p = path.Path(c)
        particle_pixl = p.contains_points(coord)
        particle_pixl = np.reshape(particle_pixl, (Ny, Nx)) - 0.0
        particle = particle_pixl*Density
         
        img = np.maximum(img, particle)
        all_part_mask = np.maximum(all_part_mask, particle_pixl)
        
        sub_mask_details['image_id'].append(image_id+1)
        sub_mask_details['category_id'].append(2)
        sub_mask_details['pixl_arr'].append(particle_pixl)
        sub_mask_details['id'].append(annotation_id)
        annotation_id += 1
        
    return img, all_part_mask

def process_directories(root):
    directory = os.getcwd() + root
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)
    image_path = directory + "images/"
    mask_path = directory + "masks/"
    os.mkdir(image_path)
    os.mkdir(mask_path)
    return image_path, mask_path

annotation_id = 1
sub_mask_details = dict()
sub_mask_details["image_id"] = []
sub_mask_details["category_id"] = []
sub_mask_details["id"] = []
sub_mask_details["pixl_arr"] = [] 

def validate_args(args):
    assert args.count > 0, 'count must be greater than 0'
    assert args.Image_width >= 64, 'width must be greater than 64 and equal to height'
    assert args.Image_height >= 64, 'height must be greater than 64 and equal to width'
    assert args.Image_width == args.Image_height, 'image width and hight must be same'
    assert args.ncomp > 0, 'ncomp must be greater than 0'
    assert args.blend >= 0 and args.blend <= 1, 'blend must be within 0 to 1'

def create_images_and_mask(args):
    Nx = args.Image_width
    Ny = args.Image_height
    hm_img = args.count
    num = args.ncomp
    mix = args.blend
    sigma = 0.7
    
    nfiber = np.int(mix*num)
    npart = num - nfiber
    coord = np.transpose(get_coord(Nx, Ny))
    
    root = args.output_dir
    image_path, mask_path = process_directories(root)
    
    print(f'Generating {hm_img} images and corresponding mask......')
    for image_id in tqdm(range(hm_img)): # idx changed to image_id      
        image = np.random.uniform(low=0, high=0.4, size=(Ny,Nx))
        C = np.random.randint(Nx-20, size=(num,2))
        
        fiber_img, fiber_mask = fiber(image_id, nfiber, coord, C[:nfiber], image, sigma, Nx, Ny)
        part_img, part_mask = particles(image_id, npart, coord, C[-npart:], fiber_img, sigma, Nx, Ny)
        
        image = part_img
        mask = np.maximum(fiber_mask, part_mask)
        
        mask = gaussian_filter(mask, sigma)
        image = gaussian_filter(image, sigma)
        
        mask_rescaled = (255.0 / mask.max() * (mask - mask.min())).astype(np.uint8)
        image_rescaled = (255.0 / image.max() * (image - image.min())).astype(np.uint8)
        
        mask = Image.fromarray(mask_rescaled)
        mask.save(mask_path + str(image_id+1).zfill(5) +".png")
        image = Image.fromarray(image_rescaled)
        image.save(image_path + str(image_id+1).zfill(5) +".png")
        
    print(f'successfully generated {hm_img} images and their masks to folder: \n{image_path} and \n{mask_path}')

def create_coco_json(args):
    global sub_mask_details
    Nx = args.Image_width
    Ny = args.Image_height
    hm_img = args.count
    root = args.output_dir
    
    should_continue = input('would you like to create dataset annotation json? (y/n): ').lower()
    if should_continue != 'y' and should_continue != 'yes':
        quit()
    else:
        annotations = []
        #for index in range(hm_img):
        for index in range(len(sub_mask_details["id"])):
            segmentation, bbox, area = get_segmentation(sub_mask_details['pixl_arr'][index])
            annotation = get_mask_annotation(segmentation, 0, sub_mask_details['image_id'][index], sub_mask_details['category_id'][index], sub_mask_details['id'][index], bbox, area)
            annotations.append(annotation)
        
        images = []
        for index in range(hm_img):
            image_key = create_images(index, Ny, Nx)
            images.append(image_key) 
            
        info = create_info()
        licenses = []
        licenses.append(create_license())
        categories = create_categories(2)
        
        master_obj = {
                'info': info,
                'licenses': licenses,
                'images': images,
                'annotations': annotations,
                'categories': categories
                }
        
        import json   
        output_path = os.getcwd() + root + 'mycoco_instances.json'
        with open(output_path, 'w+') as output_file:
            output_file.write(json.dumps(master_obj, indent=4))
            
        print(f'Annotation successfully written to file:\n{output_path}')

def main(args):
    validate_args(args)
    create_images_and_mask(args)
    create_coco_json(args)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Artificial SEM Images")
    parser.add_argument("--Image_width", type=int, required=True, help="Width of the image")
    parser.add_argument("--Image_height", type=int, required=True, help="height of the image")
    parser.add_argument("--count", type=int, required=True, help="Number of images to be generated")
    parser.add_argument("--ncomp", type=int, required=True, help="Number of particles/fiber in a image")
    parser.add_argument("--blend", type=float, required=True, help="percentage of fiber (0 to 1)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save all the images (e.g. '/dataset/')")
    
    args = parser.parse_args()
    
    main(args)