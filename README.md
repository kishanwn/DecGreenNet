## Learning Green’s function efficiently using low-rank approximation

A novel approach to learn the Green's function with low-rank decomposition is proposed with the DecGreenNet models. 


Parameterizing the Green's function 

$$
u_{\theta_1}(x;g) =  \sum_{x' \in S_{\Omega}} G_{\theta_1}(x,x')g(x') 
$$

$$
u_{\gamma_1,\gamma_2}(x;g) =   \frac{|\Omega|}{|S_{\Omega}|} \sum_{y \in  S_{\Omega}} {\color{blue} \sum_{i=1}^{R} F_{\gamma_1}(x)_iH_{\gamma_2}(y)_i g(y) }  
 = \frac{|\Omega|}{|S_{\Omega}|} F_{\gamma_1}(x)^{\top} \left[ \sum_{y \in S_{\Omega}} H_{\gamma_2}(y) g(y)\right]
$$

### Requirements
- Python
- Pytorch

### Publication

Wimalawarne, K., Suzuki, T. & Langer, S. Learning Green’s function efficiently using low-rank approximations. Mach Learn 114, 214 (2025)



  






