# Image_with_annotation
This program is written to generate artificial SEM (Scanning Electronic Microscopic) images containing two types of fillers (short stright fibers and particles).
It can also generate the COCO compatible .JSON annotation files to segment the fillers
You can generate any number of images with any number of fillers.

Run the following command command from the terminal:
  python simulate_image_with_annotation.py --Image_width 256 --Image_height 256 --count 1000 --ncomp 50 --blend 0.5 --output_dir "/folder_name/"

Here,
--Image_width: Width of the image
--Image_height: Height of the image
--count: Number of images you want ot generate
--ncomp: Number of fillers you want to embed in each images
--blend: Mixing percentage of fibers and particles. 0 means all are particles and 1 means all are fibers
--output_dir: The folder where the output will be saved

After completing the image generation, you can also create annotation .JSON file (which is compatible with COCO annotation).
Type "y"/"yes" when you are asked after the image generation. 
Follow the instruction as appeared in the command terminal.

Finally, you will ended up with two folders (images and their correspoding mask) and one .JSON annotation file
