# Car Face Images Generator Using VAE and Web App 

By Matthew Smithey, Rosie Nguyen, and Saloni Tak
##1. Introduction

According to Bloomberg's report, cats drive almost 15 percent of all Web traffic. There are also about 30 million Google searches per month for the search term “cat".

Pew Research Center reports that animals (especially cats) are among the most popular subjects in YouTube videos. One upload from December 2016 which shows a 10 minute compilation of the funniest cat videos has racked up 68 million views.

Fun Fact 

Grumpy cat is probably the biggest current Internet animal star; she's valued at around $1 million.

​​Grumpy Cat earns her money. She has a grueling appearance schedule, a hard driving agent and an owner who works with her full time. She even has to deal with younger cats gunning for her job. No wonder she's grumpy.

So why the internet cat video obsession?

Here’s what the study’s found: Watching cat videos was associated with reduced stress, sadness, and anxiety for survey participants. In fact, the mood boost from a quick cat video sesh was so great that it overpowered any possible guilt participants felt about procrastinating. 

Stress overload can cloud your decision making skills and make work feel more challenging, so it’s not surprising that about 56 percent of employees report that their stress and anxiety negatively impacts their workspace performance, according to the Anxiety and Depression Association of America (ADAA)

In times of stress, a quick timeout can help clear your head and possibly help you work more efficiently afterwards. Diving into the internet’s endless gallery of cat videos might benefit both your health and work performance.
Problem Description

Since there is not enough cat content on the internet for cat lovers. We developed a Cat Image generator using following:
- Web App
- Generative Adversarial Network (GAN) 
- Variational Autoencoders (VAEs) 

Variational Autoencoders and Web App gave better results so we only talk about those in this blog. However, we have attached the .ipynb file of the Generative Adversarial Network. 

##2. Data Description

Applied Variational Autoencoders to two different datasets sourced from the Kaggle community.

Dataset images include cat body and background so the output wasn’t good.
Dataset with 15,747 images of cat’s face.
https://www.google.com/url?q=https://www.kaggle.com/spandan2/cats-faces-64x64-for-generative-models&sa=D&source=docs&ust=1640048125163919&usg=AOvVaw2L2XYv4CD0wyetSr_QsV_J

##3. Variational Autoencoders (VAE) Model
Model Description and Architecture
In machine learning, a variational autoencoder, also known as VAE, is the artificial neural network architecture introduced by Diederik P Kingma and Max Welling, belonging to the families of probabilistic graphical models and variational Bayesian methods.

It is often associated with the autoencoder model because of its architectural affinity, but there are significant differences both in the goal and in the mathematical formulation. Variational autoencoders are meant to compress the input information into a constrained multivariate latent distribution (encoding) to reconstruct it as accurately as possible (decoding). 

VAE effectiveness has been proven in producing highly realistic pieces of content of various kinds, such as images, texts and sounds.

VAE Neural Net Architecture
The encoder and decoder half of a traditional autoencoder simply looks symmetrical. The encoder maps the input into the code. The latent space or a bottleneck in the network forces a compressed knowledge representation of the original input and the decoder maps the code to a reconstruction of the input. 


Traditional autoencoder


On the other hand, we see the encoder part of VAE is slightly longer than its decoder with the presence of mu and sigma layers, which are the parameters of a probability distribution, presenting the mean and variance of a Gaussian. This approach produces a continuous, structured latent space, which is useful for image generation.



Variational Autoencoder (VAE)

Model Workflow

The figure below shows the model workflow





Let’s get into the code
Import libraries and Load the dataset
In this VAE model, we mainly use Tensorflow, Keras and Matplotlib libraries. The model is ran on Jupyter notebook and loads data from the local folder. 

Data Preprocessing
All cat face images are in different sizes so the first thing of the data preprocessing is resizing all cat face images to the same size: 64x64. Then we created the preprocess function in which we decoded the JPEG- encoded images to tensors of type uint8 then converted tensors to float 32. After that, we did some resizing and scaling that is called “Normalization”. Lastly, we reshaped the images to three dimensions for reperating RGB layers.










Training dataset


Loading one batch at a time can help us avoid loading all images at once and get out of memory errors. There are 124 batches.

Build and train the model
Build the encoder
The input shape of the encoder is the shape of the image (64,64,3). We used the activation function called LeakReLu because it can avoid the dying ReLU problem.
The dying ReLU problem is a serious issue that causes the model to get stuck and never let it improve. 









Build the sample layer


This is the model summary after adding decoder and sample layer


Build the decoder
The first layer is a normal dense layer which will take input from the latent vector. We used batch normalization after every layer. For the last transpose layer we have three feature matrices representing the red green and blue channel of the generated image.




Summary of the decoder


Combine encoder and decoder to make VAE


The summary of the VAE model



Loss Function
VAEs train by maximizing the evidence lower bound (ELBO) on the marginal log-likelihood:

In practice, optimize the single sample Monte Carlo estimate of this expectation:

Where z  is sampled from q(z|x)





Implementation
We will train VAE model for 100 epochs and for every 10th step save the result of whatever model learns and print summary about what is the current loss for a particular epoch and step
In this step, we also save some random images of every epoch so we can see how the images improve in every epoch.



   Output visualization
Build function to save images while learning. The function will create output_path in the directory if it does not exist. Then we take the images saved from the training the model step and create multiple subplots each 10 steps. The subplot is in 5x5 grids which means we need 25 images for each subplot. The subplot finally were saved in jpg format and named following the format of “Epoch_{:04d}_step_{:04d}.jpg” so epoch 1, step 120 will be named as “Epoch_0001_step_0120.jpg”.

The cat face images in different epochs

        
From left to right
Epoch: 1 - Step: 30 - MSE loss: 0.065924965 - KL loss: 0.17227072
Epoch: 5 - Step: 30 - MSE loss: 0.020455679 - KL loss: 0.18409342
Epoch: 10 - Step: 30 - MSE loss: 0.016867327 - KL loss: 0.24211958

       

From left to right
Epoch: 15 - Step: 120 - MSE loss: 0.013828359 - KL loss: 0.1805006
Epoch: 17 - Step: 120 - MSE loss: 0.011577732 - KL loss: 0.19423902
Epoch: 19 - Step: 100 - MSE loss: 0.012144949 - KL loss: 0.19519672

Cat Generator Timelapse using cv2


GIF of cat faces generated from the VAE model


Challenges and Potential Improvements
Overall, the output is pretty good after 19 training epochs. We can clearly see cat faces. However, the images are blurry and not realistic enough. The reason why we only trained 19 epochs is because the VAE model trained too slow and the model got an OOM error which means the GPU is running out of memory after the 19th epoch. Therefore, we couldn't continue training the model. 
Some potential improvements for the above challenges are increasing the batch size, trying to reduce the complexity of the model by reducing the number of layers and nodes in Conv2D. My laptop’s RAM is 8GB, if I want to try more epochs, perhaps, I should train the model in a laptop with higher RAM.
Web App
At the time of writing this blog, we are still in the process of developing a web application for this model. We’ll take a look at our current status, as well as ways in which we plan on further developing the app. 
Current Status 
We are currently using a placeholder image generator that was produced using a simple GAN model. The code for the GAN model was based on code taken from off the web. We slightly modified the code for our own purposes, but it is mostly not our own, so we did not include it here in the report. However, we have included a link to the site we used for reference in the footnotes. 

Google Collab 
Using a GAN model as a placeholder, we were able to run an image generator in Google Collab. The placeholder model was created on 500 epochs, so the detail of the output is relatively underdeveloped. However, you can vaguely make out the image of a cat. It looks more like impressionistic art than a photograph, but at least it’s producing output that approaches our desired output. 
plt.imshow(gen_imgs[0, :, :, 0], cmap='gray') 


StreamLit
We also ran the app in a web browser using StreamLit. Unfortunately, StreamLit does not allow the use of plotting images in the same way that Collab and other environments do. We tried converting the numpy array to an image using PLT, but it produced an image that looks nothing like the image we were able to create using the plotter. 

img = Image.fromarray(gen_imgs[0, :, :, 0], 'L')
big_img = img.resize((500,500))
st.image(big_img)


Steps for Further Development 
There are several steps for further developing the app. First, we would like to create a new image generator using our original VAE model. This takes a lot of time to train. A more powerful processor could do it faster as well. We would take this new saved model and run it in the web app in place of the placeholder model that we are currently using. We would do this in both Collab and StreamLit. 
The next step for further development would be to find a way to convert the numpy array into an image or some other data type that can be displayed on the browser using StreamLit. In the final version of the app, our goal would be to have the output in StreamLit similar to or the same as the output in Google Collab. 
