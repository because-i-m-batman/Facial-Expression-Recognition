import os
import shutil

#Making Separate Folder For Each Expression
os.mkdir('Angry')
os.mkdir('Disgust')
os.mkdir('Fear')
os.mkdir('Happy')
os.mkdir('Sad')
os.mkdir('Surprise')
os.mkdir('Neutral')

#moving images to their respective class folders

i = 0
for file in glob.glob('Path to your Output Directory where all the images ares stored after conversion from csv to jpg'):
    i+=1
    text = file.split('_')
    if(text[1] == '0.jpg'):
        shutil.move(file,'Angry/{}_0.jpg'.format(i))
    if(text[1] == '1.jpg'):
        shutil.move(file,'Disgust/{}_1.jpg'.format(i))
    if(text[1] == '2.jpg'):
        shutil.move(file,'Fear/{}_2.jpg'.format(i))
    if(text[1] == '3.jpg'):
        shutil.move(file,'Happy/{}_3.jpg'.format(i))
    if(text[1] == '4.jpg'):
        shutil.move(file,'Sad/{}_4.jpg'.format(i))
    if(text[1] == '5.jpg'):
        shutil.move(file,'Surprise/{}_5.jpg'.format(i))
    if(text[1] == '6.jpg'):
        shutil.move(file,'Neutral/{}_6.jpg'.format(i))