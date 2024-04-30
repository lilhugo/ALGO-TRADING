# Implementation for the Algorithmic Trading Course. 

This work is based on 'Hawkes model for price and trades high-frequency dynamics' by Emmanuel Bacry and Jean-François Muzy. 

We will try to realise all the numerical results proposed in this paper using Python.

1) Point Process

We simulated point processes using the Thinning method developped in Y. Ogata, 'On lewis’ simulation method for point processes', Ieee Transactions On Information Theory, 27 (1981), pp. 23–31.

2) Correlation

3) Kernel estimation

We used the kernel estimation based on the Nystrom Method for Fredholm Equation. 
We first created a function to estimate the g_t from 1000000s simulated Hawkes Trade event (Point Process Part).
We computed the conditionnal expectation of the introductory equation, and the inverse of mean of the interval time. The difference of both of them is the g_t function for our time discretization.


