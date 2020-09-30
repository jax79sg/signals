We randomly sample time segments from the output stream of each simulation, and store them in an output vector. A commonly used tensor notation for Keras, Theano, and TensorFlow which we use here is that of a 4D real float32 vector, taking the form 
```
Nexamples × Nchannels × Dim1 × Dim2.
```
 In this case we have Nexamples examples from the datasteam, each consisting of 128 complex floating point time samples. We treat this as `Nchannels = 1`, a representation which commonly is used for RGBA values in imagery, `Dim1 = 2` holding our I and Q channels, and `Dim2 = 128` holding our time dimension.

Since radio domain operations are typically considered in complex baseband representation, which is not currently well suited for for many operations in ML toolboxes such as Theano and Tensorflow. We leverage the `2-wide I/Q Dim1 to hold these in-phase and quadrature components in step with one another as a 2x128 vector`. Since automatic differentiation environments for complex valued neural networks are not yet sufficiently mature, and a ”complex convolutional layer” can obtain much of the benefit within this representation, we believe this representation is sufficient for the time being.
