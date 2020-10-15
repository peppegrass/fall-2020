#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
# Problem set 7 solutions
# Written by Tyler Ransom
# Commented by Giuseppe Grasso
#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#

using Random
using LinearAlgebra
using Statistics
using Distributions
using Optim
using DataFrames
using CSV
using HTTP
using GLM
using FreqTables
using ForwardDiff
using LineSearches
using SMM
cd("/Users/peppegrass/Documents/GitHub/fall-2020/ProblemSets/PS6-ccp/") # GG: sets working directory
pwd() ## GG: prints working directory
readdir() # GG: equivalent to -ls- to see elements of working directory

function allwrap()

    #=
    GG: In this problem set, we will practice estimating models by
    Generalized Method of Moments (GMM) and Simulated Method of Moments (SMM).
    =#

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1 - GMM for simple regression model
    # data & model are SAME AS PS2, QUESTION 2
    #
    # 1. Estimate the linear regression model from Question 2 of Problem Set 2 by GMM.
    # Write down the moment function as in slide #8 of the Lecture 9 slide deck and use Optim for estimation.
    # Use the N × N Identity matrix as your weighting matrix. Check your answer using the closed-form matrix formula for the OLS estimator.
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = log.(df.wage)

    function ols_gmm(β, X, y) # GG: GMM function
        K = size(X,2) # GG: don't really needed for GMM, unless for computing RMSE in step below
        N = size(y,1) # GG: don't really needed for GMM, unless for computing RMSE in step below
        ŷ = X*β
        g = y .- ŷ # GG: force y and y_hat to be as close as possible
        J = g'*I*g # GG: GMM objective function
        return J
    end

    # we can also estimate the RMSE of the regression by ADDING ANOTHER MOMENT (and assuming homoskedasticity)
    function ols_gmm_with_σ(θ, X, y)
        K = size(X,2)
        N = size(y,1)
        β = θ[1:end-1]
        σ = θ[end]
        g = y .- X*β
        g = vcat(g,( (N-1)/(N-K) )*var(g) .- σ^2) # this builds on the assumption that MSE: SSR/(N-K) = σ² # GG: performing DOFs corrections
        J = g'*I*g
        return J
    end

    β_hat_gmm = optimize(b -> ols_gmm(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(β_hat_gmm.minimizer)
    println("ans: ",X\y) # GG: X\y gives the least squares; cool uh!?
    β_hat_gmm = optimize(b -> ols_gmm_with_σ(b, X, y), rand(size(X,2)+1), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(β_hat_gmm.minimizer)
    println("ans: ",vcat(X\y,sqrt( sum( (y .- X*β_hat_gmm.minimizer[1:end-1]).^2 )/( size(y,1)-size(X,2) ) )))

    # GG: of course ols_gmm(b, X, y) and ols_gmm_with_σ(b, X, y) yield same results but latter with MSE

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2 - GMM for multinomial logit model
    # data & model are SAME AS PS2, QUESTION 5
    #
    # 2. Estimate the multinomial logit model from Question 5 of Problem Set 2 by the following means:
    #   (a) Maximum likelihood (i.e. re-run your code [or mine] from Question 5 of Problem Set 2)
    #   (b) GMM with the MLE estimates as starting values. Your g object should be a vector of dimension N ×J where
    #   N is the number of rows of the X matrix and J is the dimension of the choice set. Each element, g should equal d − P,
    #   where d and P are “stacked” vectors of dimension N×J
    #   (c) GMM with random starting values
    # Compare your estimates from part (b) and (c). Is the objective function globally concave?
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7

    #df = @transform(df, white = race.==1)
    df[:,:white] = df.race .== 1
    X = [ones(size(df,1),1) df.age df.white df.collgrad]
    y = df.occupation

    # get starting values for mlogit from series of binary logits
    svals = zeros(size(X,2),7)
    for j=1:7
        tempname = Symbol(string("occ",j))
        df[:,tempname] = df.occupation.==j
    end
    # GG: this is to derive some "reasonable starting values" for a multinomial logit model
    # GG: what do we mean for "reasonable starting values". You don't wanna have neither a vector of 0 nor a vector of random numbers
    # GG: So what starting values for a Multinomial logit. Estimate a bunch of binary logits and then...
    svals[:,1] = coef(lm(@formula(occ1 ~ age + white + collgrad), df))
    svals[:,2] = coef(lm(@formula(occ2 ~ age + white + collgrad), df))
    svals[:,3] = coef(lm(@formula(occ3 ~ age + white + collgrad), df))
    svals[:,4] = coef(lm(@formula(occ4 ~ age + white + collgrad), df))
    svals[:,5] = coef(lm(@formula(occ5 ~ age + white + collgrad), df))
    svals[:,6] = coef(lm(@formula(occ6 ~ age + white + collgrad), df))
    svals[:,7] = coef(lm(@formula(occ7 ~ age + white + collgrad), df))
    svals = svals[:,1:6] .- svals[:,7] # GG: ... impose the identification normalization (i.e. you subtract for the first 6 estimates the one of the 7th)
    svals = svals[:] # GG: then vectorize it and use it as a starting value

    function mlogit_mle(α, X, y)

        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigα = [reshape(α,K,J-1) zeros(K)]

        P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))

        loglike = -sum( bigY.*log.(P) ) # GG: MLE objective function

        return loglike
    end

    α_zero = zeros(6*size(X,2))
    α_rand = rand(6*size(X,2))
    # α_true = [-.5120867,-.6950393,.3165616,-1.646016,-2.300697,-.7656569]
    α_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
    α_start = α_true.*rand(size(α_true))
    println(size(α_true))
    α_hat_optim = optimize(a -> mlogit_mle(a, X, y), svals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    α_hat_mle = α_hat_optim.minimizer
    println("mlogit MLE estimates: ",α_hat_mle)

    function mlogit_gmm(α, X, y)
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigα = [reshape(α,K,J-1) zeros(K)]

        P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))

        g = zeros((J-1)*K)
        for j=1:(J-1)
            for k = 1:K
                g[(j-1)*K+k] = mean( (bigY[:,j] .- P[:,j]) .* X[:,k] ) # GG: GMM (exactly-identified) objective function
                # GG: (j-1)*K moment conditions; (j-1) equations and K betas; same number of elements as number non-zero elements in bigαlpha
            end
        end
        J = N*g⋅g #GG: exactly identified; no weighting matrix
        return J
    end

    function mlogit_gmm_overid(α, X, y)
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigα = [reshape(α,K,J-1) zeros(K)]

        P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα))) # GG: same as before P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))

        g = bigY[:] .- P[:] # GG: now in over-identified case; we vectorize big Y and P with the [:] operator and subtract the two
        # GG: this will be equivalent to NONLINEAR LEAST SQUARES

        J = g'*I*g # GG: now we have weighting matrix
        return J
    end
    #α_zero = zeros(6*size(X,2))
    #α_rand = rand(6*size(X,2))
    α_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
    #α_start = α_true.*rand(size(α_true))
    #println("starting values: ",α_rand)

    # GG: here we're playing around with different optimization algorithms, starting values, and later using autodiff

    # LBFGS
    α_hat_optim = optimize(a -> mlogit_gmm_overid(a, X, y), α_true .+ .0001*rand(size(α_true)), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=1_000, show_trace=true, show_every=100))
    α_hat_gmm = α_hat_optim.minimizer
    println("LBFGS GMM minimum: ",α_hat_optim.minimum)
    println("LBFGS GMM estimates: ",α_hat_gmm)

    # use autodiff with LBFGS
    td = TwiceDifferentiable(b -> mlogit_gmm_overid(b, X, y), svals; autodiff = :forward)
    α_hat_optim_ad = optimize(td, α_true .+ .0001*rand(size(α_true)), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=10_000, show_trace=true, show_every=100))
    α_hat_gmm_ad = α_hat_optim_ad.minimizer
    println("LBFGS GMM autodiff minimum: ",α_hat_optim_ad.minimum)
    println("LBFGS GMM autodiff estimates: ",α_hat_gmm_ad)

    # use autodiff with Newton
    α_hat_optim_ad_newton = optimize(td, α_true .+ .0001*rand(size(α_true)), Newton(linesearch = BackTracking()), Optim.Options(g_tol = 1e-5, iterations=10_000, show_trace=true, show_every=100))
    α_hat_gmm_ad_newton = α_hat_optim_ad_newton.minimizer
    println("Newton GMM autodiff minimum: ",α_hat_optim_ad_newton.minimum)
    println("Newton GMM autodiff estimates: ",α_hat_gmm_ad_newton)

    # autodiff from bad starting values
    α_hat_optim_ad_fr_bad = optimize(td, α_rand.*sign.(α_true), LBFGS(), Optim.Options(g_tol = 1e-10, f_tol = 1e-10, x_tol = 1e-10, iterations=10_000, show_trace=true, show_every=50))
    α_hat_gmm_ad_fr_bad = α_hat_optim_ad_fr_bad.minimizer
    println("LBFGS GMM autodiff minimum: ",α_hat_optim_ad_fr_bad.minimum)
    println("LBFGS GMM autodiff estimates: ",α_hat_gmm_ad_fr_bad)

    # GG: putting different answers in a DataFrame and comparing them
    # GG: found out close answers; the one using bad starting value didn't get close to other solutions; some times GMM tolerance values needed to be played around with
    compare = DataFrame(mle = α_hat_mle, gmm = α_hat_gmm, gmm_ad = α_hat_gmm_ad, gmm_ad_newton = α_hat_gmm_ad_newton, gmm_ad_fr_bad = α_hat_gmm_ad_fr_bad)
    println(compare)
    # GG: objective function globally concave, but couldn't get to global maximum from random starting values, although pretty close

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3 - SIMULATE DATA from a logit model
    #
    # 3. SIMULATE A DATASET FROM A MULTINOMIAL LOGIT MODEL, and then estimate its parameter values and verify that the estimates are close to the
    # parameter values you set. That is, for a given sample size N, choice set dimension J and parameter vector β, write a function that outputs
    # data X and Y. I will let you choose N, J, β and the number of covariates in X (K), but J should be larger than 2 and K should be larger than 1.
    # If you haven’t done this before, you may want to follow these steps:
    #   (a) Generate X using a random number generator—rand() or randn().
    #   (b) Set values for β such that conformability with X and J is satisﬁed
    #   (c) Generate the N×J matrix of choice probabilities P
    #   (d) Draw the preference shocks ε as a N×1 vector of U[0,1] random numbers
    #   (e) Generate Y as follows:
    #     • Initialize Y as an N ×1 vector of 0s
    #     • Update Y i = ∑ j=1 J 1 P ik > εi  ∑k=j J  [see text of problem set]
    #   (f) An alternative way to generate choices would be to draw a N × J matrix of ε’s from a T1EV distribution.
    #   This distribution is already deﬁned in the Distributions package. Then Y i = argmax j X i β j + ε i j .
    #   I’ll show you an example of how to do that in the solutions code for this problem set.
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    function sim_logit(N=100_000,J=4) # putting "=" after the arguments here sets default values for these arguments
        # generate X, β and P
        X = hcat(ones(N),randn(N),2 .+ 2 .* randn(N))
        if J==4
            β = hcat([1, -1, 0.5],[-2, 0.5, 0.3],[0, -0.5, 2],zeros(3)) # GG: numbers that square nicely with 4
        else
            β = 2 .* rand(size(X,2),J) .- 1
        end
        P = exp.(X*β) ./ sum.(eachrow(exp.(X*β))) # GG: usual one-liner to generate multinomial ogit choice probabilities; eachrow is summation
        # draw choices
        draw = rand(N)
        Y = zeros(N) # Initializing Y as an Nx1 vector of 0s
        for j=1:J
            Ytemp = sum(P[:,j:J];dims=2) .> draw # GG: then updating it (summing across j) based on whether choice probability is above or below draw [see formula of problem set]
            Y += Ytemp # GG: for each iteration of the loop adding Ytemp to Y
        end
        return Y,X
    end
    ySim,XSim = sim_logit() # GG: it will use "default values" for N and J set in line 244 through s"et to the default operator =""
    α_hat_sim_optim = optimize(a -> mlogit_mle(a, XSim, ySim), rand(9), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    α_hat_sim_mle = α_hat_sim_optim.minimizer
    println("simulated mlogit MLE estimates: ",α_hat_sim_mle) # GG: and here are our simulated estimates; values pretty close because sample size big and SEs low

    # now with drawing from Gumbel instead (#GG: i.e T1EV)
    # GG: python does not have the nice Distributions package as in Julia
    function sim_logit_w_gumbel(N=100_000,J=4) # putting = here sets default values for these arguments
        # generate X and β (No P needed)
        X = hcat(ones(N),randn(N),2 .+ 2 .* randn(N))
        if J==4
            β = hcat([1, -1, 0.5],[-2, 0.5, 0.3],[0, -0.5, 2],zeros(3))
        else
            β = 2 .* rand(size(X,2),J) .- 1
        end
        # draw choices
        ϵ = rand(Gumbel(0,1),N,J) # GG: using Julia's Distributions package; Gumbel(0,1) is "standard" T1EV
        Y = argmax.(eachrow(X*β .+ ϵ)) # GG: getting argmax of eachrow; eachrow allows avoiding to have to write a loop; tells Julia to do argmax computation row-wise and find column j maximizer
        # GG: that's just another way of deriving the Y; the loop in the previous function is just as fine
        return Y,X
    end
    ySim,XSim = sim_logit_w_gumbel()
    α_hat_sim_optim = optimize(a -> mlogit_mle(a, XSim, ySim), rand(9), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    α_hat_sim_mle = α_hat_sim_optim.minimizer
    println("simulated mlogit MLE estimates: ",α_hat_sim_mle)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 4 - run SMM.parallelNormal()
    #
    # 4. Use SMM.jl to run the example code on slide #21 of the Lecture 9 slide deck.
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # GG: SMM.jl has a nice count-down that Optim does not have
    MA = SMM.parallelNormal()
    dc = SMM.history(MA.chains[1])
    dc = dc[dc[:accepted].==true, :]
    println(describe(dc)) # GG: can verify you get the mean of the MVN distribution

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 5 - estimate mlogit by SMM
    # GG: Wanna use SIMULATED METHOD OF MOMENTS (using code of Q3) to ESTIMATE RESULT OF Q2
    #
    # 5. Use your code from Question 3 to estimate the multinomial logit model from Question 2 using SMM and the code example from slide #18 of
    # the Lecture 9 slide deck.
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # GG: X and y of Q2
    X = [ones(size(df,1),1) df.age df.white df.collgrad]
    y = df.occupation
    # GG: chunk below was commented out by Tyler beacuse the set of moment conditions used were just not working out for him
    #function mlogit_smm(α, X, y, D)
    #    K = size(X,2)
    #    J = length(unique(y))
    #    N = length(y)
    #    bigY = zeros(N,J)
    #    bigỸ = zeros(N,J)
    #    for j=1:J
    #        bigY[:,j] = y.==j
    #    end
    #    bigα = [reshape(α,K,J-1) zeros(K)]

    #    T = promote_type(eltype(α),eltype(X))
    #    gmodel = zeros(T,(J-1)*K,D)
    #    gdata  = zeros(T,(J-1)*K)

    #    # data moments
    #    for j=1:(J-1)
    #        gdata[(j-1)*K+1] = mean( (bigY[:,j]) )
    #        for k = 2:K
    #            gdata[(j-1)*K+k] = cov( bigY[:,j], X[:,k] )
    #        end
    #    end

    #    Random.seed!(1234)
    #    # simulated model moments
    #    for d=1:D
    #        # draw choices
    #        ε = rand(Gumbel(0,1),N,J)
    #        ỹ = argmax.(eachrow(X*bigα .+ ε))
    #        for j=1:J
    #            bigỸ[:,j] = ỹ.==j
    #        end
    #        # data moments
    #        for j=1:(J-1)
    #            gmodel[(j-1)*K+1,d] = mean( (bigỸ[:,j]) )
    #            for k = 2:K
    #                gmodel[(j-1)*K+k,d] = cov( bigỸ[:,j], X[:,k] )
    #            end
    #        end
    #    end

    #    # minimize squared difference between data and moments
    #    g = mean(gmodel .- gdata; dims=2)
    #    J = N*g⋅g
    #    return J
    #

    # GG: the function below for the SIMULATED METHOD OF MOMENTS is the one from the SLIDES OF LECTURE 9 (mind the SCROLLER; gotta use HTML and not PDF version)
    function mlogit_smm_overid(α, X, y, D) # GG: D is the number of draws
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        bigỸ = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j # GG: bigY is our real data that we're trying to MATCH
        end
        bigα = [reshape(α,K,J-1) zeros(K)] # GG: bigα is our parameter vector that we're gonna try to guess value of, so that we get values so that simulated data match actual ones as closely as possible

        Random.seed!(1234) # GG: cardinal rule, set up the seed; don't forget that!
        for d=1:D
            # draw choices
            ε = rand(Gumbel(0,1),N,J) # GG: if we change the distribution of ε to a multivariate normal, we can do MNP without having to do any Monte Carlo integration. We don't need to have a closed form solution for this.
            # GG: and if we want to do GEV models, we need to specify nest-specific and alternative-specific component; just two errors but same principle.
            ỹ = argmax.(eachrow(X*bigα .+ ε)) # GG: we're goona draw ỹ through ε
            for j=1:J
                bigỸ[:,j] .+= (ỹ.==j)*(1/D) # GG: bigỹ is our SIMULATED data
                # GG: we set the jth column of bigỸ to a dummy if ỹ is equal to j
            end
        end

        g = bigY[:] .- bigỸ[:] # GG: exactly like g in Question 2; very same principle

        J = g'*I*g
        return J
    end
    println(size(X,2))
    println(length(unique(y))-1)
    println("size of svals: ",size(α_true .+ .0001*rand(size(α_true))))
    bigα = [reshape(α_true,size(X,2),length(unique(y))-1) zeros(size(X,2))]
    println("size of bigα: ",size(bigα))
    td = TwiceDifferentiable(th -> mlogit_smm_overid(th, X, y, 2_000), α_true .+ .0001*rand(size(α_true)); autodiff = :forward)
    α_hat_smm_overid = optimize(td, α_true .+ .0001*rand(size(α_true)), LBFGS(), Optim.Options(g_tol=1e-8, x_tol=1e-8, f_tol=1e-8, iterations=100_000, show_trace=true, show_every=5))
    println(α_hat_smm_overid.minimizer)
    return nothing
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::
@time allwrap()
