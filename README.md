# Car Face Images Generator Using VAE and Web App 

By Matthew Smithey, Rosie Nguyen, and Saloni Tak
## 1. Introduction

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

## 2. Data Description

Applied Variational Autoencoders to two different datasets sourced from the Kaggle community.

Dataset images include cat body and background so the output wasn’t good.
Dataset with 15,747 images of cat’s face.
https://www.google.com/url?q=https://www.kaggle.com/spandan2/cats-faces-64x64-for-generative-models&sa=D&source=docs&ust=1640048125163919&usg=AOvVaw2L2XYv4CD0wyetSr_QsV_J

## 3. Variational Autoencoders (VAE) Model
### Challenges and Potential Improvements
Overall, the output is pretty good after 19 training epochs. We can clearly see cat faces. However, the images are blurry and not realistic enough. The reason why we only trained 19 epochs is because the VAE model trained too slow and the model got an OOM error which means the GPU is running out of memory after the 19th epoch. Therefore, we couldn't continue training the model. 
Some potential improvements for the above challenges are increasing the batch size, trying to reduce the complexity of the model by reducing the number of layers and nodes in Conv2D. My laptop’s RAM is 8GB, if I want to try more epochs, perhaps, I should train the model in a laptop with higher RAM.
## 4. Web App
At the time of writing this blog, we are still in the process of developing a web application for this model. We’ll take a look at our current status, as well as ways in which we plan on further developing the app. 
Current Status 
We are currently using a placeholder image generator that was produced using a simple GAN model. The code for the GAN model was based on code taken from off the web. We slightly modified the code for our own purposes, but it is mostly not our own, so we did not include it here in the report. However, we have included a link to the site we used for reference in the footnotes. 

### Google Collab 
Using a GAN model as a placeholder, we were able to run an image generator in Google Collab. The placeholder model was created on 500 epochs, so the detail of the output is relatively underdeveloped. However, you can vaguely make out the image of a cat. It looks more like impressionistic art than a photograph, but at least it’s producing output that approaches our desired output. 

### StreamLit
We also ran the app in a web browser using StreamLit. Unfortunately, StreamLit does not allow the use of plotting images in the same way that Collab and other environments do. We tried converting the numpy array to an image using PLT, but it produced an image that looks nothing like the image we were able to create using the plotter. 

### Steps for Further Development 
There are several steps for further developing the app. First, we would like to create a new image generator using our original VAE model. This takes a lot of time to train. A more powerful processor could do it faster as well. We would take this new saved model and run it in the web app in place of the placeholder model that we are currently using. We would do this in both Collab and StreamLit. 
The next step for further development would be to find a way to convert the numpy array into an image or some other data type that can be displayed on the browser using StreamLit. In the final version of the app, our goal would be to have the output in StreamLit similar to or the same as the output in Google Collab. 
