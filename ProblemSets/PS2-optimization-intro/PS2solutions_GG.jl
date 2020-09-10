#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
# Problem set 2 solutions
# Written by Tyler Ransom
# Commented by Giuseppe Grasso
#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#

using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using CSV
using HTTP
using GLM
using FreqTables
using ForwardDiff # for bonus at the very end
cd("/Users/peppegrass/Documents/GitHub/fall-2020/ProblemSets/PS1-julia-intro/") # GG: sets working directory
pwd() ## GG: prints working directory
readdir() # GG: equivalent to -ls- to see elements of working directory

function allwrap()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1
    # Basic [# GG: NON-LINEAR] optimization in Julia
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    #=
    GG: We will use Julia’s Optim package, which is a function MINIMIZER.
    Thus, if we want to ﬁnd the maximum of f(x), we need to minimize −f(x).
    GG: Run code below in REPL to see how Optim works
    =#

    f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2 # GG: original objective function
    minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2 # GG: -(obj fct)
    startval = rand(1)   # random starting value
    #=
    GG: The Optim package provides a function called optimize(). This function requires three inputs:
    the objective function, a starting value, and an optimization algorithm.
    We will not get too deep into optimization algorithms in this course, but for now just use LBFGS().
    =#
    result = optimize(minusf, startval, BFGS()) # GG: BFGS is a gradient-based algorithm: objective function needs to be differentiable for this to work; otherwise, use different algorithm
    println(result)

    #=
    GG: Look at the measures of convergence in the result:
    3 different criteria when doing non-linear optimization. Metaphor: climing a mountain and looking at ALTITUDE & LATITUDE/LONGITUDE
    1. Look at g(x): Most intuitive and common one: is the gradient 0 [necessary condition: FOC]?
    2. Look at |f(x) - f(x')|: Since the last iteration of this procedure (with the first iteration being originally based on the starting value), how much has the FUNCTION VALUE changed (i.e. has the ALTITUDE changed) between the previous or current iteration? Is there evidence for the slope not being at zero?
    3. Look at |x - x'|: How far did I step from the last iteration to this iteration, i.e. how did the value x (current) change wrt to x' (previous)? How much did LATITUDE/LONGITUDE change
        # GG: gradient 1. is the most important and also hardest to satisfy
        Tolerance values/thresholds can be set for gradient (gtol/gval), function (ftol/fval) and value (xtol/xval); e.g. e-06 as opposed to e-08
    =#

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2
    # Now that we’re familiar with how Optim’s optimize() function works, lets try it on some real-world data.
    # Speciﬁcally, let’s use Optim to compute OLS estimates of a simple linear regression using actual data.
    # The process for passing data to Optim can be tricky, so it will be helpful to go through this example.
    # First, let’s import and set up the data. Note that you will need to put the URL all on one line when executing this code in Julia.
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv" # GG: setting URL
    df = CSV.read(HTTP.get(url).body) # GG: reading CSV from URL
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1] # GG: subset of variables that will be used in regression
    y = df.married.==1 # GG: copying variable married from DataFrame df into array y
    # GG: By applying OLS to a binary choice model we are estimating a Linear Probability Model (LPM)

    # GG: OLS is minimizer of Sum of Squared Errors (SSR) so we write down the objective function
    function ols(beta, X, y) # GG: argument you are optimizing over has to come first in Optim; others (data) can come later
        ssr = (y.-X*beta)'*(y.-X*beta) # GG: one way of writing it (.- is called loop fusion; should be the most efficient)
        #= ssr = dot(y.-X*beta,y.-X*beta) Julia function dot() that takes dot product (an alternative) =#
        #= ssr = [sum[y[i]-X[i,:]*beta]^2 for i in 1:length(y)] (yet another alternative by comprehension; might contain mistakes; pseudo-code) =#
        #=  ssr = 0
            for i=1:length(y)
                ssr += (y[i]-X[i,:]*beta)^2
            end
        (yet another alternative using a loop; += is an operator to do iterations)
        =#
        return ssr
    end

    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    #= GG:
    - In exercise 1 we simply used optimize(minusf, ., .) because we had a univariate function of x
    Now here we got 3 arguments and we need to tell Julia that we are optimizing over b by using the following synthax
    optimize(b -> ols(b, X, y), ...)
    - rand(size(X,2)) are starting value which must have the same dimension as the parameter vector (or dim covariate vector including the constant)
    - LBFGS() is the algorithm chosen
    - Setting a few options through Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true)
        - tolerance value for gradient; capping number of iterations at 100k; show_trace=true prints out in the REPL what's happening at every step
    =#

    println(beta_hat_ols.minimizer) # GG: prints the output

    # GG: Since OLS has a closed form solution, we can check that this worked in a few different ways:
    # 1. by evaluating the known OLS closed form
    bols = inv(X'*X)*X'*y
    println(bols)
    # 2. by using the GLS package; very similar to R
    df.white = df.race.==1 # GG: Creating a dummy white for race==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df) # GG: like in R, once you tell where data is coming from, you can just use varnames stata-style
    println(bols_lm)
    # GG: It can be easily verified that all three solutions are the same

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3
    # Use Optim to estimate the logit likelihood. Some things to keep in mind:
    #   To maximize the likelihood, you will need to pass Optim the negative of the likelihood function
    #   The likelihood function is included in the Lecture 4 slides
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    function logit(alpha, X, y)

        P = exp.(X*alpha)./(1 .+ exp.(X*alpha))

        loglike = -sum( (y.==1).*log.(P) .+ (y.==0).*log.(1 .- P) )

        return loglike
    end
    alpha_hat_optim = optimize(a -> logit(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(alpha_hat_optim.minimizer)


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 4
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    alpha_hat_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    println(alpha_hat_glm)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 5
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    freqtable(df, :occupation) # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation) # problem solved

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    function mlogit(alpha, X, y)

        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end

        P = num./repeat(dem,1,J)

        loglike = -sum( bigY.*log.(P) )

        return loglike
    end

    alpha_zero = zeros(6*size(X,2))
    alpha_rand = rand(6*size(X,2))
    alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
    alpha_start = alpha_true.*rand(size(alpha_true))
    println(size(alpha_true))
    alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_mle = alpha_hat_optim.minimizer
    println(alpha_hat_mle)


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # bonus: how to get standard errors?
    # need to obtain the hessian of the obj fun
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # first, we need to slightly modify our objective function
    function mlogit_for_h(alpha, X, y)

        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

        T = promote_type(eltype(X),eltype(alpha)) # this line is new
        num   = zeros(T,N,J)                      # this line is new
        dem   = zeros(T,N)                        # this line is new
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end

        P = num./repeat(dem,1,J)

        loglike = -sum( bigY.*log.(P) )

        return loglike
    end

    # declare that the objective function is twice differentiable
    td = TwiceDifferentiable(b -> mlogit_for_h(b, X, y), alpha_start; autodiff = :forward)
    # run the optimizer
    alpha_hat_optim_ad = optimize(td, alpha_zero, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_mle_ad = alpha_hat_optim_ad.minimizer
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, alpha_hat_mle_ad)
    # standard errors = sqrt(diag(inv(H))) [usually it's -H but we've already multiplied the obj fun by -1]
    alpha_hat_mle_ad_se = sqrt.(diag(inv(H)))
    println([alpha_hat_mle_ad alpha_hat_mle_ad_se]) # these standard errors match Stata

    return nothing
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::
allwrap()
