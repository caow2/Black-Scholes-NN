# Black-Scholes-NN
Research project for Rowan's Summer Undergraduate Research Program (2018). The research objective is to develop a neural network to solve the Black-Scholes equation. <br>

The resources used are referenced below:
* https://pdfs.semanticscholar.org/5ebb/e0b1a3d7a2431bbb25d6dfeec7ed6954d633.pdf
* http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=E4F4F0CC997A64C0B822DB4978CE344F?doi=10.1.1.336.8359&rep=rep1&type=pdf
* https://pdfs.semanticscholar.org/1de0/33feff65e6a9052ba18b7d472c2129145b1a.pdf
<br>

The neural network is implemented using TensorFlow. In the current implementation, the neural net attempts to solve a sample PDE that is similar to the BSE.

The data used to train the neural network is generated uniformly based on the sample problem. 80% of the data is chosen randomly for training and 20% is used for validation on the trained model.

The given sample equations and boundary conditions were: <br>
u(x, 10) = g(x) = (e^x) / 5 <br>
u(0, t) = f0(t) = (e^-0.01(10-t)) * (1/5)        //f0 is f subscript 0 <br>
u(2, t) = f1(t) = (e^-0.01(10-t)) * ((e^2)/5)    //f1 is f subscript 1 <br>

Dirichlet BCs: <br>
f0(10) = g(0) <br>
f1(10) = g(2) <br>
               
The main goal for a neural network approach to solve the BSE is efficiency - methods such as the finite elements method require alot of computation time and memory, especially if the problem is scaled into higher dimensions.
<br>
**Neumann Boundary conditions were not considered, only Dirichlet.**
      
