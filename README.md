## Learning Green’s function efficiently using low-rank approximation

A novel approach to learn the Green's function with low-rank decomposition is proposed. Applying low rank decomposition to the Green's function exables seperating the computation by the input and Monte-Carlo summations.    


#### Parameterizing the Green's function 

$$
u_{\theta_1}(x;g) =  \sum_{x' \in S_{\Omega}} G_{\theta_1}(x,x')g(x') 
$$

#### DecGreenNet

$$
u_{\gamma_1,\gamma_2}(x;g) =   \frac{|\Omega|}{|S_{\Omega}|} \sum_{y \in  S_{\Omega}}  \sum_{i=1}^{R} F_{\gamma_1}(x)_iH_{\gamma_2}(y)_i g(y)   
 = \frac{|\Omega|}{|S_{\Omega}|} F_{\gamma_1}(x)^{\top} \left[ \sum_{y \in S_{\Omega}} H_{\gamma_2}(y) g(y)\right]
$$


### Requirements
- Python
- Pytorch

### Code execution

```bash
python train_poi2D_multi_nl_plot.py --num_quad 100
```

### Publication

Wimalawarne, K., Suzuki, T. & Langer, S. Learning Green’s function efficiently using low-rank approximations. Mach Learn 114, 214 (2025)



  






