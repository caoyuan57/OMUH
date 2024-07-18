# OMUH
Object-guided Multi-granularity Unsupervised Hashing for Large-scale Image Retrieval.

# Main Dependencies
-python 3.9.12
-pytorch 1.13.1
-torchvision 0.14.1
-numpy 1.21.5

# Data
The VOC2012, FLICKR25K and MSCOCO datasets are all publicly available.
You can download VOC2012 at https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
You can download FLICKR25K at http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip
You can download MSCOCO at http://images.cocodataset.org/zips/train2014.zip   http://images.cocodataset.org/zips/val2014.zip
# Training
Fisrt, generate the object pseudo-Label for each dataset.
$ python object_label.py 
````
Then, train the hashing model.
````
$ python OMUH.py
````
The relevant parameters can be adjusted in the file's config.