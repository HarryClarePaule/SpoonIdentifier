# SpoonIdentifier

Weclome to the very handy Spoon identifier programme!

This programme consist of multiple scripts:
1. spoon_identifier.py - in this script the spoon_identifier_model.h5 has been created and trained using a data set of images
2. main.py - in this script the model has been fed a series of validation images to test the accuracy of the model
3. app.py - in this script a simple flask web application has been developed to allow someone to upload a picture for the model to predict if it is a spoon or not

notes - this model is fairly accurate, however it does struggle if you upload photos of other utensils like knives or forks, it needs further training in order to learn that a knife is in fact not a sppon.

Hope you enjoy!
