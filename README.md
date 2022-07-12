
## Grace: Graph-based representation learning for fault localization

Grace is a coverage-based fault localization approach with graph-based representation. This repository includes data and code about the technique.


### Execution
runtotal.py is  main entry file.  Using ``python runtotal.py subname`` (e.g., python runtotal.py Lang) would execute the run.py, sum.py, watch.py respectively.

* run.py is for each buggy version of each project, which is repeatedly executed in runtotal.py. 
* sum.py merges the results for all the buggy version of one project.
* watch.py prints the results. 
* Model.py is about the model.
* Dataset.py is about the dataset. 

The final results are logged in the directory ``result_final_XXXX`` while the third line is the number of Top-1 Value. 


### Environment
```
PyTorch: V1.7.1
OS: Ubuntu 16.04.6 LTS
```



### Dataset
The preprocessed dataset could be download in [link](https://drive.google.com/drive/folders/1QH_Y9fKaNrwQCT6hvAH9-73PBQQ3a4hL?usp=sharing). Please ensure that the path to  these .pkl file is correctly set in the code. 




