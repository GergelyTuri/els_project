# Aliza Hacking
## Summer 2023

### Video Conversion/Analysis Instructions

* In the Turi_lab shared drive, in Data → PTSD_project → ELS2_PTSD → shockboxes, there are a series of folders each containing a males folder and a femalesI folder
  
* In each males and femalesI folder, there is an embedded folder either entitled males_avis or femalesI_avis respectively, which contains .avi format versions of videos in the males and femalesI folders that have been converted from .ffii to .avi format
    * The notebook that completes this conversion is also in the shared drive, and is called videoconversionavi.ipynb
        * If you get a “broken pipe” error in running this notebook, make sure that the account you’re using to run the notebook has permission to move files within the shared drive - this is likely the issue
    * There is no femalesI_avis folder for recall02 because the females folder in recall02 had no .ffii files to begin with

* Beyond this, the colab_demo_SuperAnimal_GT.ipynb colab notebook in the colab folder can use the code from deeplabcut zoo to analyze a whole folder of .avis at once rather than having to go through one video at a time, which is what the colab zoo code that comes straight off the [zoo website does](http://www.mackenziemathislab.org/dlc-modelzoo)
    * Currently, this runs effectively to the point of producing h5 and csv analysis files, but fails when you try to export/generate labeled videos
    * Essentially, the main change that is made from the notebook from zoo itself to the one above is that the path to the “file” is instead the path to the folder containing the videos in brackets so that it’s presented as a list

* Otherwise, the config files produced by zoo and by normal deeplabcut look very similar - the only significant differences are the body parts and the cropping dimensions, both of which should be relatively easy to alter
    * You could theoretically edit the zoo configs at the top to list the file path as a path to a folder with videos rather than a path to a singular video
    * So far, the labeled videos I’ve looked at that are produced from zoo are generally decent when the dots do show up, but there will periods of a few seconds where all or most of the dots will just disappear off the screen
        * Side note: when using Zoo, make sure to use the function that allows you to decrease the dot size (try size 4), otherwise it becomes very difficult to really evaluate the tracking by eye in the labeled videos

* For anyone running deeplabcut on a Mac, it can be kind of tricky to get it to open from the terminal once you have everything downloaded, but here is the code that has worked for me: 
    * conda activate deeplabcut
    * pythonw
    * import deeplabcut
    * deeplabcut.launch_dlc()  

* Ideally, if anyone could figure out a way to input the resnet training file that is spit out of zoo and input it successfully into deeplabcut for network refinement, that would be the goal


