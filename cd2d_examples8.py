from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import deepxde as dde
from deepxde.backend import tf
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
## This code is fine
epsilon = 1e-8

def main():

    # Testing it for example 8
    def pde(x, u):
        theta = -math.pi/3
        du_xx = dde.grad.hessian(u, x,i=0, j=0)
        du_yy =dde.grad.hessian(u, x,i=0, j=1)
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_y = dde.grad.jacobian(u, x, i=0, j=1)
        return -epsilon*(du_xx+du_yy) + math.cos(theta)*du_x +math.sin(theta)*du_y

    def boundary(x, on_boundary):
        return on_boundary

    def func(xe):
        u_exact = []
        # print(xe)
        theta = -math.pi/3

        # def zero(x, on_boundary):
        #     return on_boundary and (x[0] > (1. - DOLFIN_EPS) or  x[1] < 0.7)
        # def one (x, on_boundary):
        #     return on_boundary and (x[0] < (1. - DOLFIN_EPS) and x[1] >= 0.7)
        for x in xe:
            if (x[0] > 0.0001 and x[0] < 0.9999 and x[1] > 0.0001 and x[1] < 0.9999):
                if (x[1] > (0.7+ x[0]*math.sin(theta)/math.cos(theta))):
                    u_exact.append(1.0)
                else:
                    u_exact.append(0.0)
            elif ((x[0] < 1.e-10 and x[1] > 0.7) or (x[1] > 0.999999 and x[0] < 0.999999)):
                u_exact.append(1.0)
            # elif boundary(x,on_boundary) and (x[0] > (1. - 1.e-14) or  x[1] < 0.7):
            #     u_exact.append(0.0)
            # elif boundary(x,on_boundary) and (x[0] < (1. - 1.e-14) and x[1] >= 0.7):
            #     u_exact.append(0.0)
            else:
                u_exact.append(0.0)
            
        return np.array(u_exact).reshape(-1,1)
        
    # Convert it to a 2D geometry   
        
    geom = dde.geometry.Rectangle([0,0],[1,1])
    bc = dde.DirichletBC(geom, func, boundary)
    data = dde.data.PDE(geom, pde, bc, num_domain = 2400, num_boundary =240, solution=func, num_test=3200)

    layer_size = [2] + [12] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2error"])
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
    
    # # Plot PDE residual
    # model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)
    # x = geom.uniform_points(5, True)
    # y = model.predict(x, operator=pde)
    # plt.figure()
    # plt.plot(x, y)
    # plt.xlabel("x")
    # plt.ylabel("PDE residual")
    # plt.show()


if __name__ == "__main__":
    main()
