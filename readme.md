# SIGGRAPH 2021: Discovering Diverse Athletic Jumping Strategies

[project page](https://arpspoof.github.io/project/jump/jump.html)

[paper](https://www.cs.sfu.ca/~kkyin/papers/Jump.pdf)

[demo video](https://www.youtube.com/watch?v=DAhZ6oDoNHg&t=1s)

[![image_0032](https://user-images.githubusercontent.com/37004963/115977003-4c299300-a528-11eb-813c-7eab6efd5515.png)](wiki/GetStarted.md)

### Prerequisites

#### Important Notes
We suspect there are bugs in linux gcc > 9.2 or kernel > 5.3 or our code somehow is not compatible with that. Our code has large numerical errors from unknown source given the new C++ compiler. Please use older versions of C++ compiler or test the project on Windows.

#### C++ Setup
This project has C++ components. There is a ```cmake``` project inside ```Kinematic``` folder. We have setup the CMake project so that it can be built on both linux and Windows. Use ```cmake```, ```cmake-gui``` or visual studio to build the project. It requires ```eigen``` library.

#### Python Setup
Install the Python requirements listed in ```requirements.txt```. The version shouldn't matter. You should be safe to install the latest versions of these packages.

#### Rendering Setup
To visualize training results, please set up our simulation renderer.
- Clone and follow build instructions in [UnityKinematics](https://github.com/arpspoof/UnityKinematics). This is a flexible networking utility that will send raw simulation geometry data to Unity for rendering purpose. 
- Copy ```[UnityKinematics build folder]/pyUnityRenderer``` to this root project folder.
- Here's a sample Unity project called [SimRenderer](https://github.com/arpspoof/SimRenderer) in which you can render the scenes for this project. Clone SimRenderer outside this project folder.
- After building UnityKinematics, copy ```[UnityKinematics build folder]/Assets/Scripts/API``` to ```SimRenderer/Assets/Scripts```. Start Unity, load SimRenderer project and it's ready to use.


### Training P-VAE
We have included a pre-trained model in ```results/vae/models/13dim.pth```. If you would like to retrain the model, run the following:
```
python train_pose_vae.py
```
This will generate the new model in ```results/vae/test**/test.pth```. Copy the ```.pth``` file and the associated ```.pth.norm.npy``` file into ```results/vae/models```. Change ```presets/default/vae/vae.yaml``` under the ```model``` key to use your new model. 

### Train Run-ups
```
python train.py runup
```
Modify ```presets/custom/runup.yaml``` to change parts of the target take-off features. Refer to Appendix A in the paper to see reference parameters. 

After training, run
```
python once.py runup no_render results/runup***/checkpoint_2000.tar
```
to generate take-off state file in ```npy``` format used to train take-off controller.

### Train Jumpers
Open ```presets/custom/jump.yaml```, change ```env.highjump.initial_state``` to the path to the generated take-off state file, like ```results/runup***/checkpoint_2000.tar.npy```. Then change ```env.highjump.wall_rotation``` to specify the wall orientation (in degrees). Refer to Appendix A in the paper to see reference parameters (note that we use radians in the paper). Run
```
python train.py jump
```
to start training.  

Start the provided SimRenderer (in Unity), enter play mode, the run
```
python evaluate.py jump results/jump***/checkpoint_***.tar
```
to evaluate the visualize the motion at any time. Note that ```env.highjump.initial_wall_height``` must be set to the training height at the time of this checkpoint for correct evaluation. Training height information is available through training logs, available both in the console and through tensorboard logs. You can start tensorboard through
```
python -m tensorboard.main --bind_all --port xx --logdir results/jump***/
```
