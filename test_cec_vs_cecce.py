def solve_dare(A,B,Q,R):
    P = dare(A,B,Q,R)
    K = -inv(R + (B.T)@P@B)@B.T@P@A
    return (P,K)

if __name__ == "__main__":
    # required packages
    from scipy.linalg import solve_discrete_are as dare
    from scipy.linalg import LinAlgError, inv, pinv, norm, eigvalsh
    from numpy.linalg import det
    from numpy.random import multivariate_normal
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(10)
    ## Problem parameters

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
    K = K_init

    # optimal controller
    (P_opt, K_opt) = solve_dare(A,B,Q,R)

    # gaussian noise
    cov = np.eye(d_x)
    mean = np.zeros(d_x)

    # number of simulations and horizon
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


        # Scenario -- optimal
        x = np.zeros([T+1, d_x]) # states
        u = np.zeros([T+1, d_u]) # controls
        c = np.zeros(T)          # costs

        # Scenario -- No doubling trick
        x1 = np.zeros([T+1, d_x]) # states
        u1 = np.zeros([T+1, d_u]) # controls
        c1 = np.zeros(T)          # costs
        # Scenario -- doubling trick
        x2 = np.zeros([T+1, d_x]) # states
        u2 = np.zeros([T+1, d_u]) # controls
        c2 = np.zeros(T)          # costs


        # Scenario 1
        V1 = np.eye(d)
        S1 = np.zeros([d_x, d])
        L1 = 0
        K1 = K_init

        # CECCE hyper parameters
        K_2 = K_init
        delta = 0.05
        safe = False
        conf = 10**10 # initial confidence is infinite
        k = 1

        # rate functions
        h_t = 1
        f_t = 1
        g_t = 1
        sigma_t = 1

        # Scenario 1 -- conditions for playing certainty equivalence
        ce1 = False # certainty equivalence
        bg1 = False # bounded gain
        se1 = False # sufficient exploration



        # control
        for t in range(T):
            #if t%1000 == 0:
            #    print('simulation '+ str(n) +': iteration '+ str(t))
            h_t = t+1
            f_t = (t+1)**2
            g_t = (t+1)**3
            # LSE
            # Build controller
            # Build

            if t > 1:
                # Scenario I - CEC - Least squares estimator
                z1 = np.hstack((x1[t-1],u1[t-1])).reshape(d, 1)
                V1 = V1 + z1@(z1.T)
                S1 = S1 + (x1[t].reshape(d_x, 1))@(z1.T)
                (A1, B1) = np.split(S1@pinv(V1), d_x ,axis=1)

                # Scenario I - CEC - Solution to Ricatti equations
                try:
                    (P1,K1) = solve_dare(A1, B1, Q, R)
                except LinAlgError:
                    K1 = K_init;


                # Scenario I - CECCE
                if np.ceil(np.log(t)/np.log(2)) == np.log(t)/np.log(2):
                    k = np.log(t)/np.log(2) - 1
                    # Least squares estimation
                    Z = np.hstack((x2[int(t/2):t], u2[int(t/2):t]))
                    Y = x2[int(t/2)+1:t+1]
                    (A2, B2) = np.split((Y.T@Z)@pinv((Z.T)@Z), d_x ,axis=1)
                    # Solution to Ricatti equations
                    try:
                        (P2,K2) = solve_dare(A2, B2, Q, R)
                    except LinAlgError:
                        K2 = K_init
                    # Initial phase - CECCE
                    if not safe and np.min(eigvalsh(Z.T@Z)) >=1:
                        conf = 6*(1/np.min(eigvalsh(Z.T@Z)))*(d*np.log(5) + np.log(4*(k**2)*det(Z.T@Z)/delta))
                        C_safe = 54*(norm(P2)**5)
                        print('iteration '+ str(t) + ': '+str(1/conf) +' > '+ str(9*(C_safe**2)) + ' ?')
                        if  1/conf > 9*(C_safe**2):
                            print(t)
                            safe = True
                            A_safe = A2
                            B_safe = B2
                            K_safe = K2
                            sigma2 = np.sqrt(d_x)*np.power(norm(P2),9/2)*np.max(1, norm(B2))*np.sqrt(np.log(norm(P2)/delta))




            # Scenario I - CEC - warm up conditions
            bg1 = (norm(K1) <= h_t)
            se1 = (np.min(eigvalsh(V1)) > np.power(t+1, (1/4)))
            L1 = L1 + x1[t]@x1[t]
            if L1 > d_x*g_t:
                ce1 = False;
            elif L1 < d_x*f_t:
                ce1 = True;


            # Scenario I - optimally controlled system
            u[t] = x[t]@(K_opt.T)
            c[t] = norm(x[t])**2 + norm(u[t])**2
            x[t+1] = x[t]@A + u[t]@B + w[t];


            # Scenario I-CEC -- Controlled system
            if bg1 and se1 and ce1:
                u1[t] = x1[t]@(K1.T) + np.power((d_x/(t+1)), 1/4)*v[t]
            else:
                u1[t] = x1[t]@(K_init.T) + np.power((d_x/(t+1)), 1/4)*v[t]
            c1[t] = norm(x1[t])**2 + norm(u1[t])**2
            x1[t+1] = x1[t]@A + u1[t]@B + w[t];
            regret1[n, t+1] = regret1[n, t] + c1[t] - c[t]


            # Scenario I-CECCE -- Controlled system
            if not safe:
                u2[t] = x2[t]@(K_init.T) + v[t]
            elif norm(K2 - K_safe) > conf:
                u2[t] = x2[t]@(K_safe.T) + np.sqrt(sigma2)*np.power(k, 1/4)*v[t]
            else:
                u2[t] = x2[t]@(K2.T) + np.sqrt(sigma2)*np.power(k, 1/4)*v[t]
            c2[t] = norm(x2[t])**2 + norm(u2[t])**2
            x2[t+1] = x2[t]@A + u2[t]@B + w[t]
            regret2[n, t+1] = regret2[n, t] + c2[t] - c[t]



    # Scenario I -CEC
    avg_regret1 = np.mean(regret1[:,1:], axis=0)
    std_regret1 = np.std(regret1[:,1:], axis=0, ddof=1)/np.sqrt(N)

    # Scenario I- CECCE
    avg_regret2 = np.mean(regret2[:,1:], axis=0)
    std_regret2 = np.std(regret2[:,1:], axis=0, ddof=1)/np.sqrt(N)

    steps = np.arange(T)

    plt.plot(steps, avg_regret1, label='CEC($\mathbb{N}$)')
    plt.plot(steps, avg_regret2, label='CECCE')
    plt.fill_between(x=range(T), y1=avg_regret1+std_regret1,  y2=avg_regret1-std_regret1, alpha=0.3)
    plt.fill_between(x=range(T), y1=avg_regret2+std_regret2,  y2=avg_regret2-std_regret2, alpha=0.3)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.xlabel('Iterations')
    plt.ylabel('Regret')

    plt.show()
