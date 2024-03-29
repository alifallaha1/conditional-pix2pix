# Conditional Pix2Pix
[![Python 3.10.13](https://img.shields.io/badge/python-3.10.13-blue.svg)](https://www.python.org/downloads/release/python-31013/)
[![PyTorch 2.1.2](https://img.shields.io/badge/pytorch-2.1.2-ee4c2c.svg)](https://pytorch.org/)
![Built with Love](https://img.shields.io/badge/built%20with-%E2%9D%A4-red.svg)

## Overview

Conditional Pix2Pix is an innovative architecture of GAN models that integrates the Pix2Pix GAN and CGAN techniques. This fusion allows the model to generate images based on both an input image and accompanying labels. The generated image closely resembles the input image but incorporates specified changes outlined by the provided labels. This approach opens up new possibilities for image generation tasks by leveraging conditional information to guide the generation process effectively

## Example
The quality of the generated images may not be optimal due to limitations in the training data quality and duration. It's essential to carefully select high-quality training data and train the model for an adequate duration to achieve better results.


| Label | Input Image | Target Image | Generated Image | 
|-------|-------------|-----------------|------------|
| jacket | ![Input Image 1](images/input1.png) |![Target Image 1](images/real1.png) | ![Generated Image 1](images/rse1.png) | 
| jeans | ![Input Image 2](images/input2.png) |![Target Image 2](images/real2.png) | ![Generated Image 2](images/rse2.png) | 
| suit | ![Input Image 3](images/input3.png) | ![Target Image 3](images/real3.png) | ![Generated Image 3](images/res3.png) | 
| suit | ![Input Image 4](images/input4.png) |  ![Target Image 4](images/real4.png) | ![Generated Image 4](images/res4.png) |



## Getting started with the Training 
To get started with the Conditional Pix2Pix, follow these steps:

1. Clone the repository:

      ```bash
      git clone https://github.com/alifallaha1/conditional-pix2pix.git
      ```

2. Install the required dependencies. We recommend using a virtual environment:
   ```bash
         pip install -r requirements.txt
         ```

3. if you want to change the layers you can find the models in 'models.py'

4. you will find the training code and all the config parameters in the 'cpix2pix.ipynb'

## Contributing

Contributions are welcome and greatly appreciated! To contribute to this project, follow these steps:

1. Fork the repository.

2. Create a new branch:

   ```bash
   git checkout -b feature/my-feature
   ```

3. Make the desired changes and commit them:
   
   ```bash
   git commit -m "Add my feature"
   ```

4. Push to the branch:
      
   ```bash
   git push origin feature/my-feature
   ```

5. Open a pull request in the main repository.


## Contact

If you have any questions, suggestions, or feedback, please feel free to contact me:

- GitHub: [https://github.com/alifallaha1](https://github.com/alifallaha1)
- LinkedIn : [https://linkedin.com/in/ali-wael-](https://linkedin.com/in/ali-wael-)

I'm open to collaboration and look forward to hearing from you!

---
Thank you for visiting the Conditional Pix2Pix repository. I hope you find it useful and informative.
