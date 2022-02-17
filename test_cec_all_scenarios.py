def solve_dare(A,B,Q,R):
    P = dare(A,B,Q,R)
    K = -inv(R + (B.T)@P@B)@B.T@P@A
    return (P,K)

if __name__ == "__main__":
    # required packages
    from scipy.linalg import solve_discrete_are as dare
    from scipy.linalg import solve_discrete_lyapunov as lyapunov
    from scipy.linalg import LinAlgError, inv, pinv, norm, eigvalsh
    from numpy.random import multivariate_normal
    import numpy as np
    import matplotlib.pyplot as plt

    ## Problem parameters
    np.random.seed(10)

    # dimensions
    d_x = 2
    d_u = 2
    d = d_x + d_u

    # state transition and state-action transition
    A = np.array([[1.02, 0.01],
                  [0.01, 0.50]])
    B = np.array([[1.00, 0.00],
                  [0.00, 1.00]])

    # state cost matrix and action cost matrix
    Q = np.eye(d_x)
    R = np.eye(d_u)

    # Initial stabilizer
    K_init = np.array([[-0.27, -0.01],
                      [-0.01, -0.13]])

    # cost of initial controller
    P_init = lyapunov(A+B@K_init, Q+K_init.T@R@K_init)
    K = K_init

    # optimal controller
    (P_opt, K_opt) = solve_dare(A,B,Q,R)


    # gaussian noise
    cov = np.eye(d_x)
    mean = np.zeros(d_x)

    # Number of simulations and number of iterations
    N = 100
    T = 1000

    # regret
    regret1  = np.zeros([N, T+1])
    regret2  = np.zeros([N, T+1])
    regret3  = np.zeros([N, T+1])

    for n in range(N):

        if 0 == n%10:
            print('number of simulations: '+str(n))

        # horizon
        w = multivariate_normal(mean, cov, T+1) # noise
        v = multivariate_normal(mean, cov, T+1) # input perturbations


        # Scenario -- unkown A,B
        x = np.zeros([T+1, d_x]) # states
        u = np.zeros([T+1, d_u]) # controls
        c = np.zeros(T)          # costs
        # Scenario -- unkown A,B
        x1 = np.zeros([T+1, d_x]) # states
        u1 = np.zeros([T+1, d_u]) # controls
        c1 = np.zeros(T)          # costs
        # Scenario -- known B
        x2 = np.zeros([T+1, d_x]) # states
        u2 = np.zeros([T+1, d_u]) # controls
        c2 = np.zeros(T)          # costs
        # Scenario -- known A
        x3 = np.zeros([T+1, d_x]) # states
        u3 = np.zeros([T+1, d_u]) # controls
        c3 = np.zeros(T)          # costs


        # Scenario 1
        V1 = np.eye(d)
        S1 = np.zeros([d_x, d])
        L1 = 0
        K1 = K_init

        # Scenario 2
        V2 = np.eye(d_x)
        S2 = np.zeros([d_x, d_x])
        L2 = 0
        K2 = K_init

        # Scenario 3
        V3 = np.eye(d_u)
        S3 = np.zeros([d_x, d_u])
        L3 = 0
        K3 = K_init

        # rate functions
        h_t = 1
        f_t = 1
        g_t = 1
        sigma_t = 1

        # Scenario 1 -- conditions for playing certainty equivalence
        ce1 = False # certainty equivalence
        bg1 = False # bounded gain
        se1 = False # sufficient exploration

        # Scenario 2 -- conditions for playing certainty equivalence
        ce2 = False # certainty equivalence
        bg2 = False # bounded gain
        se2 = False # sufficient exploration

        # Scenario 3 -- conditions for playing certainty equivalence
        ce3 = False # certainty equivalence
        bg3 = False # bounded gain
        se3 = False # sufficient exploration

        # control
        for t in range(T):
            h_t = t+1
            f_t = (t+1)**2
            g_t = (t+1)**3
            # LSE
            # Build controller
            # Build
            if t > 0:
                # Least squares estimator
                z = np.hstack((x1[t-1],u1[t-1])).reshape(d, 1)
                V1 = V1 + z@(z.T)
                V2 = V2 + (x2[t-1].reshape(d_x,1)) @(x2[t-1].reshape(d_x,1).T)
                V3 = V3 + (u3[t-1].reshape(d_u,1)) @(u3[t-1].reshape(d_u,1).T)
                S1 = S1 + (x1[t].reshape(d_x, 1))@(z.T)
                S2 = S2 + (x2[t] -  u2[t-1]@(B.T)).reshape(d_x,1)@(x2[t-1].reshape(d_x,1).T)
                S3 = S3 + (x3[t] -  x3[t-1]@(A.T)).reshape(d_x,1)@(u3[t-1].reshape(d_u,1).T)

                # Scenario 1 - Least squares estimation of A,B
                (A1, B1) = np.split(S1@pinv(V1), d_x ,axis=1)
                # Scenario 2 - Least squares estimation of A
                A2 = S2@pinv(V2)
                # Scenario 3 - Least squares estimation of B
                B3 = S3@pinv(V3)



                # Scenario I - Solution to Ricatti equations
                try:
                    (P1,K1) = solve_dare(A1, B1, Q, R)
                    # error_K[t] = norm(K1 - K_opt)
                    # error_P[t] = np.trace(P1 - P_opt)
                except LinAlgError:
                    K1 = K_init;

                # Scenario 2 - Solution to Ricatti equations
                try:
                    (P2,K2) = solve_dare(A2, B, Q, R)
                except LinAlgError:
                    K2 = K_init;

                # Scenario 3 - Solution to Ricatti equations
                try:
                    (P3,K3) = solve_dare(A, B3, Q, R)
                except LinAlgError:
                    K1 = K_init;




            # Scenario 1 - warm up conditions
            bg1 = (norm(K1) <= h_t)
            se1 = (np.min(eigvalsh(V1)) > np.power(t+1, (1/4)))
            L1 = L1 + x1[t]@x1[t]
            if L1 > d_x*g_t:
                ce1 = False;
            elif L1 < d_x*f_t:
                ce1 = True;

            # Scenario 2 - warm up conditions
            bg2 = (norm(K2) <= h_t)
            se2 = True
            L2 = L2 + x2[t]@x2[t]
            if L2 > d_x*g_t:
                ce2 = False;
            elif L2 < d_x*f_t:
                ce2 = True;


            # Scenario 3 - warm up conditions
            bg3 = (norm(K3) <= h_t)
            se3 = (np.min(eigvalsh(V3)) > np.power(t+1, (1/2)))
            L3 = L3 + x3[t]@x3[t]
            if L3 > d_x*g_t:
                ce3 = False;
            elif L3 < d_x*f_t:
                ce3 = True;


            # optimally controlled system
            u[t] = x[t]@(K_opt.T)
            c[t] = norm(x[t])**2 + norm(u[t])**2
            x[t+1] = x[t]@A + u[t]@B + w[t];


            # Scenario 1 -- Controlled system
            if bg1 and se1 and ce1:
                u1[t] = x1[t]@(K1.T) + np.power((d_x/(t+1)), 1/4)*v[t]
            else:
                u1[t] = x1[t]@(K_init.T) + np.power((d_x/(t+1)), 1/4)*v[t]
            c1[t] = norm(x1[t])**2 + norm(u1[t])**2
            x1[t+1] = x1[t]@A + u1[t]@B + w[t];
            regret1[n, t+1] = regret1[n, t] + c1[t] - c[t] #np.trace(P_opt)

            # Scenario 2 -- Controlled system
            if bg2 and se2 and ce2:
                u2[t] = x2[t]@(K2.T)
            else:
                u2[t] = x2[t]@(K_init.T)
            c2[t] = norm(x2[t])**2 + norm(u2[t])**2
            x2[t+1] = x2[t]@A + u2[t]@B + w[t]
            regret2[n, t+1] = regret2[n, t] + c2[t] - c[t] # np.trace(P_opt)

            # Scenario 3 -- Controlled system
            if bg3 and se3 and ce3:
                u3[t] = x3[t]@(K3.T)
            else:
                u3[t] = x3[t]@(K_init.T) + v[t]
            c3[t] = norm(x3[t])**2 + norm(u3[t])**2
            x3[t+1] = x3[t]@A + u3[t]@B + w[t];
            regret3[n, t+1] = regret3[n, t] + c3[t] - c[t] # np.trace(P_opt)


    # Plotting the norm 2 of the states
    # time line
    avg_regret1 = np.mean(regret1[:,1:], axis=0)
    std_regret1 = np.std(regret1[:,1:], axis=0, ddof=1)/np.sqrt(N)

    avg_regret2 = np.mean(regret2[:,1:], axis=0)
    std_regret2 = np.std(regret2[:,1:], axis=0, ddof=1)/np.sqrt(N)

    avg_regret3 = np.mean(regret3[:,1:], axis=0)
    std_regret3 = np.std(regret3[:,1:], axis=0, ddof=1)/np.sqrt(N)

    steps = np.arange(T)
    plt.plot(steps, avg_regret1, label='Scenario I - $(A,B)$ unkown')
    plt.plot(steps, avg_regret2, label='Scenario II - $B$ known')
    plt.plot(steps, avg_regret3, label='Scenario III - $A$ known')
    plt.fill_between(x=range(T), y1=avg_regret1+std_regret1,  y2=avg_regret1-std_regret1, alpha=0.3)
    plt.fill_between(x=range(T), y1=avg_regret2+std_regret2,  y2=avg_regret2-std_regret2, alpha=0.3)
    plt.fill_between(x=range(T), y1=avg_regret3+std_regret3,  y2=avg_regret3-std_regret3, alpha=0.3)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.xlabel('Iterations')
    plt.ylabel('Regret')


    plt.show()
