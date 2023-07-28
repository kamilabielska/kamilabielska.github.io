---
layout: post
title:  "Training StyleGAN3 on Colab in 2023"
date:   2023-07-27 23:40:00 +0200
categories: stylegan3 colab gan
---
StyleGAN3 was released few years back and since then Python libraries and Colab have evolved, so simply cloning [the official repo](https://github.com/NVlabs/stylegan3) and running the code on Colab does not work anymore. Instead of guessing which python/pytorch/... version to use to avoid any errors and conflicts, it's best to replicate the original environment as closely as possible. On the StyleGAN3 repo we can find a file with conda environment, so the instinctive thing to do is set up Conda (or MiniConda, to be exact) on Colab with the right Python version and then simply update the environment.

Make sure that you have selected the GPU runtime and let's get going.


# Installing MiniConda

To install MiniConda we need to run the bash code below:
{% highlight bash %}
%%capture
%%bash

MINICONDA_INSTALLER_SCRIPT=Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX
{% endhighlight %}

We specify the Python version we want to use (see list here: `https://repo.continuum.io/miniconda`) and wait a few seconds for MiniConda to install. Then we add the new Python to the path (remember about the version) and we're set:

{% highlight python %}
import sys
sys.path.append('/usr/local/lib/python3.9/site-packages')
{% endhighlight %}

How do we know which Python version to install? In the requirements section on the offical repo we can see Python 3.8 being mentioned, however when we create the environment from the file, it installs version 3.9 (in the file itself we can see that `python >= 3.8` is specified). Version 3.8 probably would be fine, although I simply installed what the conda thought would be the right choice, so 3.9. It works, so why overthink:

install whatever version → create the environment → check the version installed in this environment → start again with the version suggested


# Installing dependencies

We could follow the advice given on the official repo and create the environment from the file and then activate it. This however poses some problems on Colab, as it will default to the base env anyway, which causes problems with the library versions and so on. Long story short, I could not get it to work properly, so instead I updated the base environment which finally made it run without any problems:

{% highlight bash %}
!git clone -q https://github.com/NVlabs/stylegan3
!conda env update -q -n base -f stylegan3/environment.yml
{% endhighlight %}

You would think.

In theory, the env file should set up everything so that the code is ready to execute, but
- right away it complained that `psutil` library is missing
- for some reason after several ticks of running without any issues, an error popped up saying that `numpy` should be downgraded in order for some `scipy` function to run correctly
- in another attempt, again in the middle of the training, there was some issue with `setuptools`, which needed to be downgraded due to some known error in newer versions

We (re)install these dependencies with the command below and hope for the best. That is, that we do not get any errors out of nowhere in the future.

{% highlight bash %}
!conda install -q -y psutil numpy==1.22.3 setuptools==58.0.4
{% endhighlight %}

If we want we can also install tensorboard and activate the extension in Colab to easily track the training process.


# Training

At this point it's just a matter of running scripts provided by the Nvidia team. To train the StyleGAN3 model on our custom dataset, we need to have all the images in a zip file or a folder and we can preprocess them using `dataset_tools.py`:

{% highlight bash %}
!python stylegan3/dataset_tool.py --source=anime.zip --dest=anime_64x64.zip \
	--transform=center-crop --resolution=64x64
{% endhighlight %}

Then we can launch training with:

{% highlight bash %}
!python stylegan3/train.py --outdir="$OUT_PATH" --data="$DATA_PATH" \
    --cfg=stylegan3-t --cbase=16384 --workers=2 --gpus=1 --aug=noaug --mirror=1 \
    --batch=$batch_size --gamma=$gamma --snap=$snap
{% endhighlight %}

and resume it if interrupted with:

{% highlight bash %}
!python stylegan3/train.py --outdir="$OUT_PATH" --data="$DATA_PATH" \
    --cfg=stylegan3-t --cbase=16384 --workers=2 --gpus=1 --aug=noaug --mirror=1 \
    --batch=$batch_size --gamma=$gamma --snap=$snap --resume="$resume_path"
{% endhighlight %}

But this is all described in detail on the official github repository. To see all of the code from this post in context, check out the `anime_stylegan3.ipynb` notebook on [my github repo](https://github.com/kamilabielska/gans).

Training does take a long time, especially if we use basic GPU provided by Colab for free and take into account playing with different hyperparameters to find the optimal ones. That's why transfer learning is worth trying out, especially because Nvidia shared some [models](https://github.com/NVlabs/stylegan3#additional-material) trained on different datasets and at different resolutions. Reusing them is a matter of providing the url to the chosen pickled model after the `--resume` flag and making sure that architecture related hyperparameters are right.

Here is a video illustrating the fine-tuning process, because it's always fun to watch:
![stylegan3 progress](https://github.com/kamilabielska/gans/blob/main/img/stylegan3_progress.gif?raw=true)

# Summary

As you can see, training StyleGAN3 on Colab is not too complicated, however setting up the environment required a fair share of googling, debugging and dealing with seemingly random errors (but what doesn't). 

