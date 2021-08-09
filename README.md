# Multi-level Space-time Registration of Growing Plants

In this project, we implement a new method for the space-time registration 
of a growing plant based on matching the plant at different geometric scales.

## Set up

The following python packages are needed to run the sample code

- pyntcloud
- tqdm
- numpy
- pandas
- open3d
- networkx
- scipy
- matplotlib
- Dijkstar
- scikit-learn
- SpharaPy
- polyscope
- robust-laplacian
- joblib
- imageio

Install these python packages by `pip install -r requirements.txt`

## Usage 

### Registration

- Unzip the data compressed in to `data/` fold.
- Run the registration pipeline with: `python3 run_registration_pipeline --type XXX --method YYY`
  
    Replace `XXX` by the target type of plant in `arabidopsis`, `tomato`, `maize`. If it's not explicitly specified, it's set by default to `tomato`.
    
    Replace `YYY` by the method. We provide two choices of method: `local_icp` and `fm`, with `local_icp` our method and `fm` the functional map method as the reference method. If it's not explicitly specified, it's set by default to `local_icp`
  
- Attention: every time rerun the registration process, please delete the existed `data/{type}/registration_result` directory
- The expected visualization results:
    
registration result of tomato:

![tomato](/imgs/tomato.png)

registration result of maize:

![maize](/imgs/maize.png)

- the running time and metric scores might not be exactly the same as presented in the paper. Since
  the scores are computed on random sampled subsets. But they should be of the same order. 

### Interpolation

- First run the registration framework to get the point-wise correspondence. Check `data/XXX/registration_result` to see if the registration work is done
- Run the interpolation pipeline with: `python3 run_interpolation --type XXX` with `XXX` the type of plant
- The interpolated point clouds will be shown frame by frame. You can save the images and generate a video with them. 
- The expected result:

<img src="/imgs/interpolation_tomato.gif" width="400" height="300" />

compared with the interpolation video produced by funtional map based method: 


<img src="/imgs/interpolation_tomato_fm.gif" width="500" height="350" />

We can see that the interpolation produced by our method is more smooth and less noisy. 