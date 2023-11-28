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
- To generate new images for custom images, copy the custom images inside the `outDomainData/images/` folder and run the following:

  ```bash
  cd outDomainData
  python3 img.py
  python3 getAttr.py
  ```
  The `img.py` script crops the image with the appropriate size.
  The `getAttr.py` asks you for the image name and all of its attributes.
  
  Running the above commands would generate lines corresponding to images and attributes in `gen.txt` copy the lines in `attr.txt` provided in `outDomainData` folder and then run the following:
  ```bash
  cd ..
  python3 generate.py --type outDomainData
  ```
  
  The generated results would get stored in the folder `./results/outDomainData`
  
- To generate new images by using random sample images from the CelebA dataset, run the following:
  ```bash
  python3 sample.py
  ```
  Give the on-screen inputs for the data folder name and number of images. The resulting images get generated with the corresponding directory structure:
  ```
  .
  ├── <folder-name>
  │   ├── images
  │   │   ├── 000001.jpg
  │   │        ...
  │   │   └── <number_of_images>.jpg
  │   └── attr.txt
  ```
  
### Evaluating the generated Images
We use two different metrics GeneratorLoss and Frechet inception distance [(FID)](https://en.wikipedia.org/wiki/Frechet_inception_distance) for comparing the performance across different GAN models. 
The GeneratorLoss for an image $x$ originally from domain a being translated to domain b is defined as $$G_{loss} = -E[D_{src}(G(x,b))] + \lambda_{1} \times E[||x - G(G(x,b),a)||] + \lambda_{2} \times E[-log(D_{cls}(b | G(x,b)]$$.   
The lower the generator loss or fid_score value, the better is the model.  
To get the scores simply run:
```bash
python3 evaluator.py
python3 fid.py
```
The commands generate corresponding files: `losses_<gan_name>.txt` and `fid_score_<gan_name>` containing the scores of different GANs.

### Files and utilities
- `stargan_generator.py`: Contains the model class and functions to generate images using StarGAN
- `attention_generator.py`: Contains the model class and functions to generate images using AttentionGan
- `attgan_generator.py`: Contains the model class of the entire ATTGAN and functions to generate images using AttGAN
- `sampleTest.py`: Randomly sample images from the CelebA dataset.  
- `generate.py`: Generate the fake images for the original images in the path provided.
- `evaluation.py`: Evaluates different models based on their Quantitative measures of fid or Generator loss 
- `fid.py`: Calculates the FID associated with the image and its corresponding translated image
- `outDomainData/getAttr.py`: Asks for attributes for a custom image from the user and writes to a file
- `outDomainData/img.py`: Used to crop custom images to fit to the CelebA training images size
 
