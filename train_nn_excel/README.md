# Train a Neural Network using Excel

View the excel sheets online

1. [Excel sheet 1](https://docs.google.com/spreadsheets/d/13d23tbpe210Y8rhmIp6GADgnqzQfl3Tf/edit#gid=1351071457)
2. [Excel sheet 2](https://1drv.ms/x/s!AjN1NGjZ4GEZvB1E9xO0UZe8ssIV?e=q5cCkW)



## Feed Forward Network (Forward Propagation)
During feed forward, we modify the value of the neurons based on the input value & the weights assigned

# Input -> Layer-1
h1=w1*i1+w2*i2
h2=w3*i1+w4*i2
out_h1=σ(h1)=1/(1+e^-h1)
out_h2=σ(h2)=1/(1+e^-h2)

# Layer-1 -> Layer-2
o1=w5*out_h1+w6*out_h2
o2=w7*out_h1+w8*out_h2
out_o1=σ(o1)=1/(1+e^-o1)
out_o2=σ(o2)=1/(1+e^-o2)

# Layer-2 -> Error/Loss (MSE Loss)
E_total=E1+E2
E1=1/2*(t1-out_o1)^2
E2=1/2*(t2-out_o2)^2

## Backward Propagation
During back propagation, the weights are modified to assist the modifications in the neuron values during it's next feedforward to reduce the Error/Loss in the next iteration

# Calculation of partial derivatives of Total Loss (E_Total) wrt each of the weights in the Network
Here, each of the weights in the network are updated basis its effect on the Total Loss (E_Total)
(We update each parameter in the network based on its effect on the Final Loss)

# Calculation of Backpropagation w.r.t weights (w5,w6,w7,w8) for 2nd layer
∂(E_total)/∂w5 = ∂(E1)/∂w5 = ∂(E1)/∂out_o1 * ∂(out_o1)/∂o1 * ∂(o1)/∂w5
∂(E1)/∂out_o1 = - (t1 - out_o1) = out_o1 - t1
∂(out_o1)/∂o1 = ∂(σ(o1))/∂o1 = out_o1 * (1 - out_o1) 
∂(o1)/∂w5 = out_h1
∂(E_total)/∂w5 = (out_o1 - t1) * out_o1 * (1 - out_o1) * out_h1
∂(E_total)/∂w6 = (out_o1 - t1) * out_o1 * (1 - out_o1) * out_h2
∂(E_total)/∂w7 = (out_o2 - t2) * out_o2 * (1 - out_o2) * out_h1
∂(E_total)/∂w8 = (out_o2 - t2) * out_o2 * (1 - out_o2) * out_h2

# Calculation of Backpropagation w.r.t weights (w1,w2,w3,w4) for 1st layer

	∂(E_total)/∂out_h1=?
	∂(E1)/∂out_h1 = ∂(E1)/∂out_o1 * ∂(out_o1)/∂o1 * ∂(o1)/∂out_h1 = (out_o1 - t1)* out_o1 * (1 - out_o1)* w5
Similarly,	∂(E2)/∂out_h1 = (out_o2 - t2)* out_o2 * (1 - out_o2)* w7
So,	∂(E_total)/∂out_h1 = ∂(E1+E2)/∂out_h1 = ∂(E1)/∂out_h1+ ∂(E2)/∂out_h1
	# (the derivative of a sum is equal to the sum of the derivatives)
	∂(E_total)/∂out_h1 = (out_o1 - t1)* out_o1 * (1 - out_o1)* w5 + (out_o2 - t2)* out_o2 * (1 - out_o2)* w7
	
Similarly,	∂(E_total)/∂out_h2 = (out_o1 - t1)* out_o1 * (1 - out_o1)* w6 + (out_o2 - t2)* out_o2 * (1 - out_o2)* w8
	
	∂(E_total)/∂w1 = ∂(E_total)/∂out_h1 * ∂(out_h1)/∂h1 * ∂(h1)/∂w1
	∂(E_total)/∂w1 = ((out_o1 - t1)* out_o1 * (1 - out_o1)* w5 + (out_o2 - t2)* out_o2 * (1 - out_o2)* w7) * (out_h1*(1-out_h1)) * i1
	
	∂(E_total)/∂w2 = ∂(E_total)/∂out_h1 * ∂(out_h1)/∂h1 * ∂(h1)/∂w2
	∂(E_total)/∂w2 = ((out_o1 - t1)* out_o1 * (1 - out_o1)* w5 + (out_o2 - t2)* out_o2 * (1 - out_o2)* w7) * (out_h1*(1-out_h1)) * i2
	
	∂(E_total)/∂w3 = ∂(E_total)/∂out_h2 * ∂(out_h2)/∂h2 * ∂(h2)/∂w3
	∂(E_total)/∂w3 = ((out_o1 - t1)* out_o1 * (1 - out_o1)* w6 + (out_o2 - t2)* out_o2 * (1 - out_o2)* w8) * (out_h2*(1-out_h2)) * i1
	
	∂(E_total)/∂w4 = ∂(E_total)/∂out_h2 * ∂(out_h2)/∂h2 * ∂(h2)/∂w4
	∂(E_total)/∂w4 = ((out_o1 - t1)* out_o1 * (1 - out_o1)* w6 + (out_o2 - t2)* out_o2 * (1 - out_o2)* w8) * (out_h2*(1-out_h2)) * i2
  
 # Updation of weights after each epoch wrt learning rate
w1=w1-lr*(∂(E_total)/∂w1)














