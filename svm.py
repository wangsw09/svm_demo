from __future__ import print_function
import sys
import numpy as np
import numpy.random as npr

def sgdStepWeight():
    i = 0.0
    while True:
        i += 1.0
        yield i

def svm(y, X, C, fig=False, eps=0.25):
    '''
    this function implement SVM by gradient descent
    argument:
        y: response
        X: input
        C: cost
        fig: whether or not plot figures, default False
    value:
        w, b: separating hyperplane f(x) = <w, x> + b
    '''

    ## --- initialization
    n, d = X.shape
    w  = np.zeros(d)
    b  = 0

    ## --- control parameters
    eta = 3e-7
    itr = 0  # iteration count
    MAX_ITER = 500
    
    ## --- if figure == True, track obj
    if fig is True:
        objTrack = np.zeros(MAX_ITER)

    ## --- objective function
    obj = C * n
    obj0 = obj / 2.0  # to pass the first loop

    # -- update objTrack
    if fig is True:
        objTrack[itr] = obj

    ## --- update iteration
    keyCond = np.maximum(0, 1 - y * (np.dot(X, w) + b))

    while np.absolute((obj - obj0)) / obj > eps / 100.0:
        obj0 = obj

        ## --- update w, b
        w = (1 - eta) * w
        for i in xrange(n):
            if keyCond[i] > 0:
                w = w + eta * C * y[i] * X[i, :]
                b = b + eta * C * y[i]

        ## --- update keyCond
        keyCond = np.maximum(0, 1 - y * (np.dot(X, w) + b))

        ## --- update objective function
        obj  = np.sum(w ** 2) / 2.0 + C * np.sum(keyCond)  # objective function

        itr += 1
        # -- check MAX_ITER
        if itr >= MAX_ITER:
            print("max iteration {0} reached, quit iteration.".format(MAX_ITER), file=sys.stderr)
            break

        # -- update objTrack
        if fig is True:
            objTrack[itr] = obj


    if fig is True:
        return (w, b, objTrack[: itr + 1])

    return (w, b)

def sgd_svm(y, X, C, fig=False, eps=0.001):
    '''
    this function implement SVM by stochastic gradient descent
    argument:
        y: response
        X: input
        C: cost
        fig: whether or not plot figures, default False
    value:
        w, b: separating hyperplane f(x) = <w, x> + b
    '''

    ## --- initialization
    n, d = X.shape
    w  = np.zeros(d)
    b  = 0

    index = npr.choice(n, size=n, replace=False)
    
    ## --- control parameters
    # eta   = 1e-4
    eta0 = 0.000002
    Delta = 0
    itr = 0  # iteration count
    MAX_ITER = n * 5
    wgt = sgdStepWeight()
    
    ## --- if figure == True, track obj
    if fig is True:
        objTrack = np.zeros(MAX_ITER)

    ## --- objective function
    obj = C * n
    obj0 = obj / 2.0  # to pass the first loop

    # -- update objTrack
    if fig is True:
        objTrack[itr] = obj

    ## --- update iteration
    keyCond = np.maximum(0, 1 - y * (np.dot(X, w) + b))
    Delta = 0.5 * Delta + 0.5 * np.absolute((obj - obj0)) / obj * 100.0
    eta = eta0 / wgt.next()

    while Delta > eps:
        obj0 = obj

        ## --- update w, b
        w = (1 - eta) * w
        if keyCond[index[itr % n]] > 0:
            w = w + eta * C * y[index[itr % n]] * X[index[itr % n], :] * n
            b = b + eta * C * y[index[itr % n]] * n

        ## --- update keyCond
        keyCond = np.maximum(0, 1 - y * (np.dot(X, w) + b))

        ## --- update objective function
        obj  = np.sum(w ** 2) / 2.0 + C * np.sum(keyCond)  # objective function

        itr += 1
        Delta = 0.5 * Delta + 0.5 * np.absolute((obj - obj0)) / obj * 100.0
        eta = eta0 / wgt.next()

        # -- check MAX_ITER
        if itr >= MAX_ITER:
            print("max iteration {0} reached, quit iteration.".format(MAX_ITER), file=sys.stderr)
            break

        # -- update objTrack
        if fig is True:
            objTrack[itr] = obj


    if fig is True:
        return (w, b, objTrack[: itr + 1])

    return (w, b)

def mnBat_svm(y, X, C, fig=False, eps=0.01):
    '''
    this function implement SVM by mini-batch stochastic gradient descent
    argument:
        y: response
        X: input
        C: cost
        fig: whether or not plot figures, default False
    value:
        w, b: separating hyperplane f(x) = <w, x> + b
    '''

    ## --- initialization
    n, d = X.shape
    w  = np.zeros(d)
    b  = 0

    index = npr.choice(n, size=n, replace=False)
    
    ## --- control parameters
    #eta   = 1e-5
    eta0 = 0.000002
    Delta = 0
    batch_size = 20
    itr = 0  # iteration count
    MAX_ITER = n / batch_size * 10
    wgt = sgdStepWeight()
    
    ## --- if figure == True, track obj
    if fig is True:
        objTrack = np.zeros(MAX_ITER)

    ## --- objective function
    obj = C * n
    obj0 = obj / 2.0  # to pass the first loop

    # -- update objTrack
    if fig is True:
        objTrack[itr] = obj

    ## --- update iteration
    keyCond = np.maximum(0, 1 - y * (np.dot(X, w) + b))
    Delta = 0.5 * Delta + 0.5 * np.absolute((obj - obj0)) / obj * 100.0
    eta = eta0 / wgt.next()

    while Delta > eps:
        obj0 = obj

        ## --- update w, b
        w = (1 - eta) * w
        for i in xrange(itr * batch_size, (itr + 1) * batch_size):
            if keyCond[index[i % n]] > 0:
                w = w + eta * C * y[index[i % n]] * X[index[i % n], :] * n / batch_size
                b = b + eta * C * y[index[i % n]] * n / batch_size

        ## --- update keyCond
        keyCond = np.maximum(0, 1 - y * (np.dot(X, w) + b))

        ## --- update objective function
        obj  = np.sum(w ** 2) / 2.0 + C * np.sum(keyCond)  # objective function

        itr += 1
        Delta = 0.5 * Delta + 0.5 * np.absolute((obj - obj0)) / obj * 100.0
        eta = eta0 / wgt.next()

        # -- check MAX_ITER
        if itr >= MAX_ITER:
            print("max iteration {0} reached, quit iteration.".format(MAX_ITER), file=sys.stderr)
            break

        # -- update objTrack
        if fig is True:
            objTrack[itr] = obj

    if fig is True:
        return (w, b, objTrack[: itr + 1])

    return (w, b)

