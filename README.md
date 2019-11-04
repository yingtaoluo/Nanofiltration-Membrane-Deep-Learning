# Nanofiltration-Membrane-Deep-Learning
It is a completed interdisciplinary research project submitted to Science Advances.  
Main contributions include:  
**Deep learning** based nanofiltration membrane performance prediction system.  
Spatial geometry-based **feature engineering & data augmentation** methods.
## Abstract  
Machine learning overfitting caused by data scarcity greatly limits the application of chemical artificial intelligence. As the data collection for chemical engineering is generally costly, here we proposed to extract the natural features of molecular structure and rationally distort it without compromising chemical characteristics to augment the dataset. The increased amount of data allows a wider range of chemical engineering projects to leverage the powerful fit of deep learning without big data at the outset. The rejection and flux predictions of polyamide nanofiltration membranes were exemplified to practice the chemical data augmentation method in deep learning and test its feasibility. Convergence of loss function indicated that the model was effectively optimized. Pearson correlation coefficient exceeding 0.80 proved a strong correlation between model predictions and real values. The success of predicting nanofiltration membrane performances is also instructive for other disciplines that were featured at molecular level.
## Author & Affiliation Statement
Ziyang Zhang1#, Yingtao Luo2#, Huawen Peng1, Yu Chen3*, Rong-Zhen Liao1*, Qiang Zhao1*  
  
1 Key Laboratory of Material Chemistry for Energy Conversion and Storage, Ministry of Education, School of Chemistry and Chemical Engineering, Huazhong University of Science and Technology, Wuhan 430074 China.  
2 School of Computer Science & Technology, Huazhong University of Science and Technology, Wuhan 430074 China.  
3 State Key Laboratory of Advanced Electromagnetic Engineering and Technology (AEET), also School of Electrical and Electronics Engineering (SEEE), Huazhong University of Science and Technology, Wuhan 430074 China.  
  
#These authors contributed equally: Ziyang Zhang, Yingtao Luo  
## Author Contact Information
Corresponding Author: Qiang Zhao (zhaoq@hust.edu.cn)  
For technical assistance please contact Yingtao Luo (yingtao.luo@columbia.edu)
## Prerequisite Environment (Recommended)
Visual Studio 2017, CUDA 9.0, cudnn 7.4.2, python 3.6, numpy 1.14.5, pytorch 1.12.0, matplotlib 3.0.3, argparse 1.4.0  
## Instruction Manual
We provide codes for neural network training in the repository.  
Because Argparse package is used, only by Command Prompt or Shell can you run the code. You could vary the variables in the Command Prompt in order to modify the program mode, such as training, plotting, etc. To help those who are not familiar with arg**, text file named "cmd instructions" has been included in the repository.  
After downloading the data, you need to modify the data path in the python file "train.py" before running the codes. 
## Data Availability
We provide ready-made data and codes for data generation in this link:  
https://pan.baidu.com/s/1_EU6w0xVxbzKk-YRX1l3PA  
