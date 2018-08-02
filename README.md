# C3D-Pytorch

This is a repository trying to implement [C3D-caffe][5] on pytorch,using models directly converted from C3D-Keras.    
And I modify the code to extract the full/partial features for videos. 
pretrained model on Sports-1m can be found [here][9].

## Requirements:

1. Have installed the Pytoch >= 0.4.0 version
2. You must have installed the following two python libs:
  - [Pillow][2]
  - [Pytorch][8]
  - ffmp
3. You must have downloaded the [UCF101][3] (Action Recognition Data Set) and [HMDB51][10].
4. Each single avi file is decoded with 25FPS (30FPS for HMDB51 dataset) in a single directory.
    - you can use the `./list/convert_video_to_images.sh` script to decode the ucf101 video files
    - run `./list/convert_video_to_images.sh .../UCF101 25` or `./list/convert_video_to_images.sh .../HMDB 30`
5. Generate {train,test}.list files in `list` directory. Each line corresponds to "image directory" and a class (zero-based). For example:
    - you can use the `./list/convert_images_to_list.sh` script to generate the {train,test}.list for the dataset
    - run `./list/convert_images_to_list.sh .../dataset_images 4`, this will generate `test.list` and `train.list` files by a factor 4 inside the root folder
6. Otherwise, you can use the standard splits provided by the dataset, and you may need to modify some codes to fit the proper path.


## Details on how to extract the datasets:
1. Assuming you have extracted C3D-pytroch to your Documents directory on a Linux system such us Ubuntu, your full path will become ~/Documents/C3D-pytroch. Use the Terminal to get to this directory

2. Now in the C3D-pytroch folder at ~/Documents/C3D-pytorch copy and paste the UCF101 or HMDB51(you may need use the 'unrar.py' to extract the HMDB51) folder here so that you get ~/Documents/C3D-pytorch/UCF101.

3. Navigate to ~/Documents/C3D-pytorch/list. It has bash files that will not run if you are not logged in as root or using sudo. To overcome this, got to ~/Documents/C3D-pytorch/list on the terminal and type chmod +x *.sh.

4. Open the file ~/Documents/C3D-pytorch/list/convert_video_to_images.sh" with any text editor. You will notice this expression (**if (( $(jot -r 1 1 $2) > 1 )); then** ) on line 31 or so . This piece of code will not run properly unless you have jot installed and this is for Linux only, well as far as I know, so I stand to be corrected. To install jot, in your terminal, type sudo apt-get install athena-jot. You can read more about it here. http://www.unixcl.com/2007/12/jot-print-sequential-or-random-data.html

5. You are now ready to generate imaged from your videos. In the terminal, still in the list directory as ~/Documents/C3D-pytroch/list, type ./convert_video_to_images.sh ~/Documents/C3D-tensorflow/UCF101 25. You can find the explanation as former mentioned.

You are now ready to generate lists from your images. In the terminal, still in the list directory as ~/Documents/C3D-pytorch/list, type ./convert_images_to_list.sh ~/Documents/C3D-tensorflow/UCF101 4 You can find the explanation as former mentioned.

Now move out of the list directory into the ~/Documents/C3D-tensorflow and run the train_c3d_ucf101.py file. If you use Python 2 you should not have problems, I guess. However, if you use python 3.5+ you my have to work on the range functions. You will have to convert them to list because in python 3+, ranges are not necessarily converted to list. So you have to do **list(range(......))**

Be sure to also have [crop_mean.npy][11](just download the `crop_mean.py`) and c3d.pickle files in the ~/Documents/C3D-pytorch directory if you choose to use the author's codes without modification. 



## Usage

1. `python fine_tune.py` will retune C3D model. The trained model will saved in `checkpoints` directory. Make sure you have created the folder.
2. `python feature_extractor_pytorch.py` will extract the full/partial features of videos. **Be careful with the path and splits**.
3. `python hash_gan.py` will train the hash models using the model you trained. Remember to change the model path. The retrival results 
will be showed on the screen and the hash code will be stored in the current path. `H_B.npy` for training and  `test.npy` for testing.
4. `python partial_recognition.py` can evaluate how the hash code perform for recognition task. Remember to change the parameters like `bits` in the script.




[1]: https://www.tensorflow.org/
[2]: http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
[3]: http://crcv.ucf.edu/data/UCF101.php
[4]: https://github.com/dutran
[5]: https://github.com/facebook/C3D
[6]: http://vlg.cs.dartmouth.edu/c3d/
[7]:https://github.com/hx173149/C3D-tensorflow
[8]:https://github.com/pytorch/pytorch
[9]:http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle 
[10]:http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
[11]:https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0