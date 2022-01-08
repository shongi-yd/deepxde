from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import deepxde as dde
from deepxde.backend import tf

epsilon = 1

## This code is fine
## Here we print two values after every epoch 
## Step {train loss at interior pts}, {train loss at boundary pts},
# {test loss at interior pts} {test loss at boundary pts}
# test metric overall
def main():
    # Testing it for example 55
    def pde(x, u):
        du_xx = dde.grad.hessian(u, x,i=0, j=0)
        du_yy = dde.grad.hessian(u, x,i=0, j=1)
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        # du_y = dde.grad.jacobian(u, x, i=0, j=1)
        return -epsilon*(du_xx+du_yy) + du_x -1.0

    def boundary(x, on_boundary):
        return np.isclose(x[0], 0.0) or np.isclose(x[0], 1.0)  or np.isclose(x[1], 0.0) or np.isclose(x[1], 1.0)

    def func(xe):
        u_exact = []
        # print(xe)
        for x in xe:
            # print(f"shape of x:{x.shape}")
            if (x[0] > 1.e-6 and x[0] < 0.999999 and x[1] > 1.e-6 and x[1] < 0.999999):
                u_exact.append(x[0])
            else:
                u_exact.append(0.0)
        return np.array(u_exact).reshape(-1,1)

    def zerofunc(x):
        return np.zeros((len(x),1), dtype=np.float32)

    geom = dde.geometry.Rectangle([0,0],[1,1])
    bc = dde.DirichletBC(geom, zerofunc, boundary)
    data = dde.data.PDE(geom, pde, [bc], num_domain = 3500, num_boundary=3500, solution=func, num_test=3200)
    
    layer_size = [2] + [12] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2error"])
    # model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    #l2error, l2 relative error

    checkpointer = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", verbose=1, save_better_only=True
    )
    # ImageMagick (https://imagemagick.org/) is required to generate the movie.
    # movie = dde.callbacks.MovieDumper(
    #     "model/movie", [-1], [1], period=200, save_spectrum=False, y_reference=func
    # )
    losshistory, train_state = model.train(
        epochs=4000, callbacks=[checkpointer]
    )

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    # y = model.predict(np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]))
    # print(y)
    
if __name__ == "__main__":
    main()
