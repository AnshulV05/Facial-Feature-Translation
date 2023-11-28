# Facial Feature Translation

The different GAN implementations utilize the following references:

**StarGAN** : <https://github.com/yunjey/stargan>  
**AttentionGAN** : <https://github.com/Ha0Tang/AttentionGAN/tree/master/AttentionGAN-v1-multi>  
**AttributeGAN** : <https://github.com/elvisyjlin/AttGAN-PyTorch> 

We have used the [**CelebA**](https://paperswithcode.com/dataset/celeba) dataset to train the StarGAN and AttentionGAN models. The download link for the dataset can be found [here](https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0).  
We have utilized the pre-trained AttributeGAN model for the image synthesis. The different model checkpoints for AttGAN can be found [here](https://drive.google.com/drive/folders/1_E5YCb4XOTZpt6KBwBzSaJdofoqPViN8).  
The specific checkpoint used for our purpose was [128_shortcut1_inject1_none](https://drive.google.com/drive/folders/1_E5YCb4XOTZpt6KBwBzSaJdofoqPViN8).

The models trained/used are collated [here](https://drive.google.com/drive/folders/15aE-ir2eCbAWT068E8RsOa2u171SHXMp).
## Code Structure 
The Model Files are modified to work with the common dataset(CelebA) with the following folder structure:
```
.
├── data
│   ├── celeba
│   │   ├── images
│   │   │   ├── 000001.jpg
│   │   │        ...
│   │   │   └── 202599.jpg
│   │   └── list_attr_celeba.txt
```
The `<gan_model>_generator.py` files contain the functions to generate images from the three different GANs using their model checkpoints.

Download the model checkpoints from the above [link](https://drive.google.com/drive/folders/15aE-ir2eCbAWT068E8RsOa2u171SHXMp).

The trained models should be saved according to the following directory structure:
```
.
├── StarGAN
│   ├── stargan
│   |   └── models
│   |       ├──200000-D.ckpt
│   |       └──200000-G.ckpt
|       ...
├── AttentionGAN
│   ├── stargan
│   |   └── models
│   |       ├──200000-D.ckpt
│   |       └──200000-G.ckpt
|       ...
├── AttGAN
│   ├── output
│   |   └── 128_shortcut1_inject1_none
│   |       ├──checkpoint
│   |       |  └──weights.49.pth
│   |       ...
|       ...
```

## Instructions to run the code
### Generating New images
- To generate new images for custom images, copy the images inside the `outDomainData/images/` folder and run the following:

  ```bash
  cd outDomainData
  python3 img.py
  python3 getAttr.py
  ```
  The `img.py` script crops the image with the appropriate size.
  The `getAttr.py` asks you for the image name and all of its attributes.
  
  Running the above commands would generate lines corresping to images and attributes in `gen.txt`.
  Copy the same in `attr.txt` provided in the same `outDomainData` folder and then cd to original project folder and run the following:
  ```bash
  python3 generate.py --type outDomainData
  ```
  
  The generated results get stored in the folder `./results/<folder_name>`
- To generate 
### Evaluating the generated Images


