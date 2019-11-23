# I found a bug about the normalization of input image, I will fix it as soon as possible.

# Pytorch_MST
Unofficial Pytorch(1.0+) implementation of ICCV 2019 paper ["Multimodal Style Transfer via Graph Cuts"](https://arxiv.org/abs/1904.04443)

This repository provides a pre-trained model for you to generate your own image given content image and style image. 

If you have any question, please feel free to contact me. (Language in English/Japanese/Chinese will be ok!)

## Requirements

- Python 3.7
- PyTorch 1.0+
- TorchVision
- Pillow
- PyMaxflow

Anaconda environment recommended here!

(optional)

- GPU environment 



## test

1. Clone this repository 

   ```bash
   git clone https://github.com/irasin/Pytorch_MST
   cd Pytorch_MST
   ```

2. Prepare your content image and style image. I provide some in the `content` and `style` and you can try to use them easily.

3. Download the pretrained model [here](https://drive.google.com/open?id=1uWnn0tHloP4wKeTIXcyQNA2MJsNRvOKA)

4. Generate the output image. A transferred output image and a NST_demo_like image will be generated.

   ```python
   python test.py -c content_image_path -s style_image_path
   ```

   ```
   optional arguments:
   -h, --help             show this help message and exit
   --content, -c          Content image path e.g. content.jpg
   --style, -s            Style image path e.g. image.jpg
   --output_name, -o      Output path for generated image, no need to add ext, e.g. out
   --n_cluster            number of clusters of k-means
   --alpha                fusion degree, should be a float or a list which length is n_cluster
   --lam                  weight of pairwise term in alpha-expansion
   --max_cycles           max_cycles of alpha-expansion
   --gpu, -g              GPU ID(nagative value indicate CPU)
   --model_state_path     pretrained model state
   ```

   If output_name is not given, it will use the combination of content image name and style image name.

------



# Result

Some results of content image will be shown here.

![image](https://github.com/irasin/Pytorch_MST/blob/master/result/avril_contrast_of_forms_demo.jpg)
![image](https://github.com/irasin/Pytorch_MST/blob/master/result/avril_scene_de_rue_demo.jpg)
![image](https://github.com/irasin/Pytorch_MST/blob/master/result/avril_picasso_self_portrait_demo.jpg)
![image](https://github.com/irasin/Pytorch_MST/blob/master/result/avril_candy_demo.jpg)
![image](https://github.com/irasin/Pytorch_MST/blob/master/result/avril_brushstrokers_demo.jpg)
![image](https://github.com/irasin/Pytorch_MST/blob/master/result/avril_asheville_demo.jpg)
![image](https://github.com/irasin/Pytorch_MST/blob/master/result/avril_antimonocromatismo_demo.jpg)
![image](https://github.com/irasin/Pytorch_MST/blob/master/result/avril_876_demo.jpg)

