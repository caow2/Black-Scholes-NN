# Black-Scholes-NN
Research project for Rowan's Summer Undergraduate Research Program (2018).
The research objective is to develop a neural network to solve the Black-Scholes equation. 
The implementation of the neural network is based on the approach to solve differential equations (Ordinary and Partial) described by Lagaris and Likas in their paper *Artificial Neural Networks for Solving Ordinary and Partial Differential Equations*. Other papers that were referenced includes Paul Wilmott's *Cliquet Options and Volatility Models* and Baymani et al's *Artificial Neural Networks Approach for Solving Stokes Problem*.
The neural network is implemented using TensorFlow. In the current implementation, the neural net solves the Black-Scholes equation with sample Dirichlet boundary/final conditions.

The given sample equations and boundary conditions were:
u(x,10) = g(x) = (e^x) / 5
u(0,t) = f0(t) = (e^-0.01(10-t)) * (1/5)        //f0 is f subscript 0
u(2,5) = f1(t) = (e^-0.01(10-t)) * ((e^2)/5)    //f1 is f subscript 1

Dirichlet BCs: f0(10) = g(0)
               f1(10) = g(2)
               
The main goal for a neural network approach to solve the BSE is efficiency - methods such as the finite elements method require alot of computation time and memory, especially if the problem is scaled into higher dimensions.

**Neumann Boundary conditions were not considered, only Dirichlet**
      
