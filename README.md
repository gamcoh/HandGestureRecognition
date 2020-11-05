# HandGestureRecognition
Hand Gesture Recognition is a deep learning project in computer vision.

## The goal
The goal is to control your computer just by making signs with your hands, for instance, if you want to play music, lower the volume, or anything else you could, just by making a certain sign.
![image](https://user-images.githubusercontent.com/18115514/98277989-4222d080-1f98-11eb-85c5-7d5ad87d6410.png)

The actionAmplifier* is just an action that will amplify another.
If you want action2 to raise the volume of your computer and you do the sign actionAmplifier1 before doing the sign action2, the volume will be raised 2 times.
The 11th action, "other", is just a picture of you without hands or gesture, because we need the model to understand when there are no hands.

## The dataset
I already have 45k images with 11 classes of signs of me and a few friends. (We need more!)
Pictures I have, have a 640x480 resolution.

## Getting started
### Installing the requirements
```shell
git clone git@github.com:gamcoh/HandGestureRecognition.git
cd HandGestureRecognition
pip install -r requirements.txt
```

Then you will need to install [TensorFlow](https://www.tensorflow.org/install)

### Generate the dataset
If you don't already have a dataset, you'll need to generate one like so:
```shell
cd utils/
python generate_dataset.py
```

You will see an OpenCV panel that will show you your webcam feed.
The script saves every frame of the live feed so you just need to worry about doing the gestures at the right time.
When you are done with one gesture, press <kbd>q</kbd> and the script will start saving the next one until you've done the 11 signs.

### Splitting the dataset
When you generated all the signs, I suggest you clean them because sometimes, the pictures are blurred or wrongly done.
Then you need to copy all the `action*` folders to `images/all`.
That way you will have:
```
HandGestureRecognition
  |
  | - images
        |
        | - all
             |
             | - action1
             | - action2
             ...
```
Now we need to split the dataset in order to have a train, test, and val folder.
To do that you just need to run the following commands:
```shell
cd utils
python split_train_test.py
```
When it's done you will see a new folder named output with the 3 folders in it, train, test, and val.

### Training the model
```shell
cd src
python train_model.py
```

### Testing the model
```shell
cd src
python run_action.py
```
Then you need to do some signs and you will see in the console if the actions are the right one.
For now, no actions will be triggered because the model is overfitting, but if you want to try the actions, you just need to remove the `continue` line in the `src/run_action.py` file.
