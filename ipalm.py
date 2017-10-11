import numpy as np
def ipalm(grads, proxs, dims, opts):
    
    ### inertial proximal alternating linearized minimization (iPALM)
    ### with fixed step size and without Lipschitz computation
    ### to solve the unconstrained optimization of form:
    ###        min_{xs} f(x0, ..., xi, ..., xn_1) + \sum_i h_i(xi)
    ### using gradient functions and proximal operators
    #
    # inputs:
    #   grads         list of functions, that evaluate gradient of f w.r.t. block i of variables
    #                    use: grad[i](xs, opts)
    #                         where xs=[x0, ..., xi, ..., xn_1]
    #   proxs         list of functions, that evaluate proximity operator of h_i
    #                    use: prox[i](v, t, opts) = min_u { t/2*||u-v||_2^2 + h_i(u)
    #                         where v has same dim as xi, t is scalar
    #   dims          list of ints, size of block xi
    #   opts           dict, optimization algorithm parameters as well as any other problem parameters
    #    opts['nit']   int, max number of iterations to perform before termination
    #    opts['tol']   float, error tolerance for termination [default: 1e-8]
    #    opts['xs0']   list/np.array, initial point for xi0 [default: random normal vector]
    #    opts['lips']  list, contains functions to compute partial Lipschitz constants
    #                    use: lip[i](xs, opts)
    #
    # outputs:
    #   xi2            final xi estimated with i = {a, b}
    
    
    #parse opts
    ks = opts.keys()
    if not 'lips' in ks:
        raise Exception('Lipschitz functions not in ''opts''')
    lips = opts['lips']
    #initial points
    xs0 = opts['xs0'] if 'xs0' in ks else [np.random.normal(size=[dims[i],]) for i in range(len(dims))]
    tol = opts['tol'] if 'tol' in ks else 1e-8
    nit = opts['nit'] if 'nit' in ks else 1e5
    
    nblocks=len(grads)
    # initialize states for inertia
    xs1=xs0
    xs2=xs0
    
    it=0 #iteration counter
    err=np.float('inf') #relative change in x. start at inf to run at least once
    #main loop
    while it < nit and err > tol:
        
        #compute inertial coefficients
        inertia = it/(it+3.0)
        
        # take proximal step in block i
        for i in range(nblocks):
            yi = xs1[i] + inertia * (xs1[i] - xs0[i])
            zi = xs1[i] + inertia * (xs1[i] - xs0[i])
            tau_i = lips[i](xs2, opts)
            xs2[i] = zi
            xs2[i] = proxs[i](yi - 1.0/tau_i * grads[i](xs2, opts), tau_i, opts)
        
        
        # compute relative error in x
        x1 = np.hstack(xs1)
        err = np.linalg.norm(x1-np.hstack(xs2))/(1e-10+np.linalg.norm(x1))
        
        #update states for inertia
        xs0 = xs1
        xs1 = xs2

        it+=1 #increment iteration counter
    return xs2