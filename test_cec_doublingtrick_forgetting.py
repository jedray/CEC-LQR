def solve_dare(A,B,Q,R):
    P = dare(A,B,Q,R)
    K = -inv(R + (B.T)@P@B)@B.T@P@A
    return (P,K)

if __name__ == "__main__":
    # required packages
    from scipy.linalg import solve_discrete_are as dare
    from scipy.linalg import solve_discrete_lyapunov as lyapunov
    from scipy.linalg import LinAlgError, inv, pinv, norm, eigvalsh
    from numpy.linalg import det
    from numpy.random import multivariate_normal
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle


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


    # initial stabilizer
    K_init = np.array([[-0.27, -0.01],
                      [-0.01, -0.13]])
    P_init = lyapunov(A+B@K_init, Q+K_init.T@R@K_init)

    K = K_init

    # optimal controller
    (P_opt, K_opt) = solve_dare(A,B,Q,R)

    # gaussian noise
    cov = np.eye(d_x)
    mean = np.zeros(d_x)

    # number of simulations and horizon
    N = 10
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


        # Scenario -- optimal
        x = np.zeros([T+1, d_x]) # states
        u = np.zeros([T+1, d_u]) # controls
        c = np.zeros(T)          # costs

        # Scenario I.(a) -- No doubling trick
        x1 = np.zeros([T+1, d_x]) # states
        u1 = np.zeros([T+1, d_u]) # controls
        c1 = np.zeros(T)          # costs
        # Scenario I.(b) -- doubling trick with forgetting
        x2 = np.zeros([T+1, d_x]) # states
        u2 = np.zeros([T+1, d_u]) # controls
        c2 = np.zeros(T)          # costs
        # Scenario I.(c) -- doubling trick with forgetting
        x3 = np.zeros([T+1, d_x]) # states
        u3 = np.zeros([T+1, d_u]) # controls
        c3 = np.zeros(T)          # costs


        # Scenario I.(a)
        V1 = np.eye(d)
        S1 = np.zeros([d_x, d])
        L1 = 0
        K1 = K_init

        # Scenario I.(b) - doubling trick with forgetting
        V2 = np.eye(d)
        S2 = np.zeros([d_x, d])
        L2 = 0
        K2 = K_init

        # Scenario I.(c) - doubling trick without forgetting
        V3 = np.eye(d)
        S3 = np.zeros([d_x, d])
        L3 = 0
        K3 = K_init

        # rate functions
        h_t = 1
        f_t = 1
        g_t = 1
        sigma_t = 1

        # Scenario I.(a) -- conditions for playing certainty equivalence
        ce1 = False # certainty equivalence
        bg1 = False # bounded gain
        se1 = False # sufficient exploration


        # Scenario I.(b) - doubling trick with forgetting
        #            -- conditions for playing certainty equivalence
        ce2 = False # certainty equivalence
        bg2 = False # bounded gain
        se2 = False # sufficient exploration
        r=2

        # Scenario I.(c) - doubling trick without forgetting
        #            -- conditions for playing certainty equivalence
        ce3 = False # certainty equivalence
        bg3 = False # bounded gain
        se3 = False # sufficient exploration


        # Scenario I.(b) -- doubling trick parameters
        gamma2 = 2
        tau2 = 2

        # Scenario I.(c) --  doubling trick parameters
        gamma3 = 2
        tau3 = 2



        # control
        for t in range(T):
            h_t = t+1
            # naive switching function
            #f_t = np.power(t+1,2)
            #g_t = np.power(t+1,3)

            # Improved switching functions
            f_t = np.log(t+1)*(t+1)
            g_t = np.power(np.log(t+1),2)*(t+1)


            if t > 1:
                # Scenario I.(a) - Least squares estimator
                z1 = np.hstack((x1[t-1],u1[t-1])).reshape(d, 1)
                V1 = V1 + z1@(z1.T)
                S1 = S1 + (x1[t].reshape(d_x, 1))@(z1.T)
                (A1, B1) = np.split(S1@pinv(V1), d_x ,axis=1)

                # Scenario I.(a) - Solution to Ricatti equations
                try:
                    (P1,K1) = solve_dare(A1, B1, Q, R)
                except LinAlgError:
                    K1 = K_init;


                # Scenario I.(b)
                if t == tau2:
                    k = tau2-1
                    # - Least squares estimator
                    Z = np.hstack((x2[int(t/2):t], u2[int(t/2):t]))
                    Y = x2[int(t/2)+1:t+1]
                    (A2, B2) = np.split((Y.T@Z)@pinv((Z.T)@Z), d_x ,axis=1)
                    V2 = Z.T@Z
                    # - Solution to Ricatti equations
                    try:
                        (P2,K2) = solve_dare(A2, B2, Q, R)
                    except LinAlgError:
                        K2 = K_init
                    tau2 = np.ceil(tau2*gamma2)

                # Scenario I.(c) - Least squares estimator
                z3 = np.hstack((x3[t-1],u3[t-1])).reshape(d, 1)
                V3 = V3 + z3@(z3.T)
                S3 = S3 + (x3[t].reshape(d_x, 1))@(z3.T)
                (A3, B3) = np.split(S3@pinv(V3), d_x ,axis=1)

                # Scenario I.(c) Solution to Ricatti equations
                if t == tau3:
                    tau3 = np.ceil(tau3*gamma3)
                    try:
                        (P3,K3) = solve_dare(A3, B3, Q, R)
                    except LinAlgError:
                        K3 = K_init;



            # Scenario I.(a) - warm up conditions
            bg1 = (norm(K1) <= h_t)
            se1 = (np.min(eigvalsh(V1)) > np.power(t+1, (1/4)))
            L1 = L1 + x1[t]@x1[t]
            if L1 > d_x*g_t:
                ce1 = False;
            elif L1 < d_x*f_t:
                ce1 = True;

            # Scenario I.(b) - warm up conditions
            bg2 = (norm(K2) <= h_t)
            se2 = (np.min(eigvalsh(V2)) > np.power(t+1, (1/4)))
            L2 = L2 + x2[t]@x2[t]
            if L2 > d_x*g_t:
                ce2 = False;
            elif L2 < d_x*f_t:
                ce2 = True;


            # Scenario I.(c) - warm up conditions
            bg3 = (norm(K3) <= h_t)
            se3 = (np.min(eigvalsh(V3)) > np.power(t+1, (1/4)))
            L3 = L3 + x3[t]@x3[t]
            if L3 > d_x*g_t:
                ce3 = False;
            elif L3 < d_x*f_t:
                ce3 = True;

            u[t] = x[t]@(K_opt.T)
            c[t] = norm(x[t])**2 + norm(u[t])**2
            x[t+1] = x[t]@A + u[t]@B + w[t];

            # Scenario I.(a) -- Controlled system
            if bg1 and se1 and ce1:
                u1[t] = x1[t]@(K1.T) + np.power((d_x/(t+1)), 1/4)*v[t]
            else:
                u1[t] = x1[t]@(K_init.T) + np.power((d_x/(t+1)), 1/4)*v[t]
            c1[t] = norm(x1[t])**2 + norm(u1[t])**2
            x1[t+1] = x1[t]@A + u1[t]@B + w[t];
            regret1[n, t+1] = regret1[n, t] + c1[t] - c[t]

            # Scenario I.(b) -- Controlled system
            if bg2 and se2 and ce2:
                u2[t] = x2[t]@(K2.T) + np.power((d_x/(t+1)), 1/4)*v[t]
            else:
                u2[t] = x2[t]@(K_init.T) + np.power((d_x/(t+1)), 1/4)*v[t]
            c2[t] = norm(x2[t])**2 + norm(u2[t])**2
            x2[t+1] = x2[t]@A + u2[t]@B + w[t];
            regret2[n, t+1] = regret2[n, t] + c2[t] - c[t]

            # Scenario I.(c) -- Controlled system
            if bg3 and se3 and ce3:
                u3[t] = x3[t]@(K3.T) + np.power((d_x/(t+1)), 1/4)*v[t]
            else:
                u3[t] = x3[t]@(K_init.T) + np.power((d_x/(t+1)), 1/4)*v[t]
            c3[t] = norm(x3[t])**2 + norm(u3[t])**2
            x3[t+1] = x3[t]@A + u3[t]@B + w[t];
            regret3[n, t+1] = regret3[n, t] + c3[t] - c[t]


    # Uncomment to save results in a pickled file
    # data = {"optimal": (P_opt, K_opt),
    #         "initial": (P_init, K_init),
    #         "numSim":N,
    #         "horizon": T,
    #         "CEC": regret1,
    #         "CEC_dt_wf": regret2,
    #         "CEC_dt": regret3}
    #
    # filename = "data_exp3_" + str(N) +"_"+ str(T)+".p"
    # pickle.dump(data, open(filename, "wb"))

    # Plotting the norm 2 of the states
    # time line
    avg_regret1 = np.mean(regret1[:,1:], axis=0)
    std_regret1 = np.std(regret1[:,1:], axis=0, ddof=1)/np.sqrt(N)

    avg_regret2 = np.mean(regret2[:,1:], axis=0)
    std_regret2 = np.std(regret2[:,1:], axis=0, ddof=1)/np.sqrt(N)

    avg_regret3 = np.mean(regret3[:,1:], axis=0)
    std_regret3 = np.std(regret3[:,1:], axis=0, ddof=1)/np.sqrt(N)

    steps = np.arange(T)

    # Plotting curves
    plt.plot(steps, avg_regret1, label='CEC($\mathbb{N}$)')
    plt.plot(steps, avg_regret3, label='CEC($\mathcal{T}_2$)')
    plt.plot(steps, avg_regret2, label='CEC($\mathcal{T}_2$) with forgetting')
    plt.fill_between(x=range(T), y1=avg_regret1+std_regret1,  y2=avg_regret1-std_regret1, alpha=0.3)
    plt.fill_between(x=range(T), y1=avg_regret3+std_regret3,  y2=avg_regret3-std_regret3, alpha=0.3)
    plt.fill_between(x=range(T), y1=avg_regret2+std_regret2,  y2=avg_regret2-std_regret2, alpha=0.3)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.xlabel('Iterations')
    plt.ylabel('Regret')
    plt.show()
