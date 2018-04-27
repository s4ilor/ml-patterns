# Hello there!
This particular tutorial shows an example of how to use TensorFlow for Poets with Kaggle dataset.

If you haven't seen the source yet - check it out! https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/ \
Much love for Google!

We're going to test it on Slavic decoration patterns. All you have to do is: download the data, type a few lines of code in shell and run the program. Simple as that!

Everything works perfectly fine on Ubuntu 16.04.

# Let's do it then!
First of all, you have to install a few things (considering you haven't done this before):

```
pip install tensorflow
pip install opencv-python
pip install kaggle
```

If you're done, head to https://www.kaggle.com and create an account. You'll use it quite frequently, trust me. After that, go to: https://www.kaggle.com/<your_name>/account and download the API Token. Then open the terminal and type:

```
git clone https://github.com/s4ilor/ml-patterns 
mv ~/Downloads/kaggle.json ~/.kaggle 
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d olgabelitskaya/traditional-decor-patterns 
mkdir ~/ml-patterns/tf_files/data ~/ml-patterns/tf_files/photos 
unzip ~/.kaggle/datasets/olgabelitskaya/traditional-decor-patterns/decor.zip -d  ~/ml-patterns/tf_files/data 
cd ~/ml-patterns/tf_files/data 
ls | grep -P "\d{2}_\d{2}_1_\d{3}.png" | xargs -d"\n" rm 
#!/usr/bin/env python 
chmod +x ~/ml-patterns/scripts/conv.py 
python ~/ml-patterns/scripts/conv.py 
ls | grep -P "\d{2}_\d{2}_2_\d{3}.png" | xargs -d"\n" rm 
cd ~/ml-patterns/tf_files/photos 
mkdir gzhel khokhoma gorodets lowickie kaszubskie iznik neglyubka 
cd ~/ml-patterns/tf_files/data 
mv 01_01*.jpg ~/ml-patterns/tf_files/photos/gzhel; 
mv 01_02*.jpg ~/ml-patterns/tf_files/photos/khokhoma; 
mv 01_03*.jpg ~/ml-patterns/tf_files/photos/gorodets; 
mv 02_04*.jpg ~/ml-patterns/tf_files/photos/lowickie; 
mv 02_07*.jpg ~/ml-patterns/tf_files/photos/kaszubskie; 
mv 03_05*.jpg ~/ml-patterns/tf_files/photos/iznik; 
mv 04_06*.jpg ~/ml-patterns/tf_files/photos/neglyubka;
rm -rf ~/ml-patterns/tf_files/data 
cd ~/ml-patterns
```

# Everything is prepared - now it's time for TensorFlow to do the work
 
Considering you HAVE NOT left the terminal, type as below:

```
IMAGE_SIZE=224
ARCHITECTURE="mobilenet_1.0_${IMAGE_SIZE}"
tensorboard --logdir tf_files/training_summaries &
python -m scripts.retrain \
   --bottleneck_dir=tf_files/bottlenecks \
   --how_many_training_steps=1000 \
   --model_dir=tf_files/models/ \
   --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
   --learning_rate=0.001 \
   --output_graph=tf_files/retrained_graph.pb \
   --output_labels=tf_files/retrained_labels.txt \
   --architecture="${ARCHITECTURE}" \
   --image_dir=tf_files/photos
```

# What the-? It doesn't work!

Let me guess, it told you:

```
"CRITICAL:tensorflow:Label neglyubka has no images in the category validation. [...] mod_index = index % len(category_list) ZeroDivisionError"
```

Of course it doesn't work. Have you checked the data at all? TensorFlow is a bit fussy when it comes to the input, and in this particular situation we don't have enough pictures in "neglyubka" folder - only 12, when TF prefers to have 20/30 at least. Say, you're an alien and you've just come to the Earth. You see a dog and a cat for a first time. It's quite hard to distinguish them later after only one encounter. 

# Two ways to go

You can either tell that data to go "#$!@ yourself" or you can download some more pictures, resize them and run the code again. I got you covered either way.

# Deleting the data

```
rm -rf ~/ml-patterns/tf_files/photos/neglyubka
```

Voila! Run the code above and you'll see the magic. Oh, and if tensorboard is troubling you, just:

```
pkill -f "tensorboard"
```

And then run the code.

# Adding more data

So you never give up. That's good... I guess. Anyway, download the NEW neglyubka folder from https://drive.google.com/open?id=1A4Tj9bydeWXHa3ygC3tPTYPXs7PIIyNA. There are more files inside which I've found on the internet and resized to 150x150px.

```
rm -rf ~/ml-patterns/tf_files/photos/neglyubka
tar xf ~/Downloads/neglyubka.tar.gz -C ~/ml-patterns/tf_files/photos
```

And run the code.

# Checking the results

After a while your network should be properly re-trained. Now it's time to check if it works.
Type into terminal:

```
python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/photos/<name_of_the_category>/<name_of_the_file.jpg>
```    

You can see that for most of the data in your directories the results are quite high - final test accuracy is somewhere near 90% (my results - 88.9% with removed neglyubka / 89.7% with new data).

# The Harder They Fall

Cool! I've got my own image classifier done in a few minutes and it works perfectly!

But... does it? Go download some more pictures, copy them into your working directory and check how good your classifier really is.
And don't worry. Low results are pretty much obvious for small amount of data and short learning time. Besides, some of these patterns ARE quite similar.


