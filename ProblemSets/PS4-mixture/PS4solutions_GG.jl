#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
# Problem set 4 solutions
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
using ForwardDiff
using LineSearches
using Distributions
cd("/Users/peppegrass/Documents/GitHub/fall-2020/ProblemSets/PS4-mixture/") # GG: sets working directory
pwd() ## GG: prints working directory
readdir() # GG: equivalent to -ls- to see elements of working directory

include("lgwt.jl")

function allwrap()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1
    # Estimate a multinomial logit (with alternative-speciﬁc covariates Z) on the following data set, which is a panel
    # form of the same data as Problem Set 3. You should be able to simply re-use your code from Problem Set 3.
    # However, I would ask that you use AUTOMATIC DIFFERENTIATION to SPEED UP your estimation, and to obtain the
    # STANDARD ERRORS of your estimates.
    # Note: this took my machine about 30 minutes to estimate using random starting values.
    # You might consider using the estimated values from Question 1 of PS3.
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code

    #=
    GG:
    The choice set is identical to that of Problem Set 3 and represents possible occupations and is structured as follows.

    1 Professional/Technical
    2 Managers/Administrators
    3 Sales
    4 Clerical/Unskilled
    5 Craftsmen
    6 Operatives
    7 Transport
    8 Other

    =#

    # GG: most of it is copy-pasted
    function mlogit_with_Z(theta, X, Z, y)

        alpha = theta[1:end-1]
        gamma = theta[end]
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

        T = promote_type(eltype(X),eltype(theta)) # GG: getting conformability right so that X and theta talk to each other
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            dem .+= num[:,j]
        end

        P = num./repeat(dem,1,J)

        loglike = -sum( bigY.*log.(P) )

        return loglike
    end
    startvals = [2*rand(7*size(X,2)).-1; .1] # GG: kinda random
    # GG: solutions to previous PS
    startvals = [ .0403744; .2439942; -1.57132; .0433254; .1468556; -2.959103; .1020574; .7473086; -4.12005; .0375628; .6884899; -3.65577; .0204543; -.3584007; -4.376929; .1074636; -.5263738; -6.199197; .1168824; -.2870554; -5.322248; 1.307477]
    td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward) # GG: declare twice-differential object
    # run the optimizer
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50)) # GG: run optimizer
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, theta_hat_mle_ad) # GG: get hessian
    theta_hat_mle_ad_se = sqrt.(diag(inv(H))) # GG: get SEs
    println([theta_hat_mle_ad theta_hat_mle_ad_se]) # these standard errors match Stata # GG: see do-file and log-file in folder

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2
    # Does the estimated coefﬁcient γˆ make more sense now than in Problem Set 3?
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # The coefficient makes much more sense than it did before. It's large, positive and significant

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3
    # Now we will estimate the mixed logit version of the model in Question 1, where γ is
    # distributed N(˜γ,σ_γ^2). Following the notes, the formula for the choice probabilities will be [see assignment text]
    # and the lig likelihood function will be [see assignment text]
    #
    # While this looks daunting, we can slightly modify the objective function from Question 1.
    # The ﬁrst step is to recognize that we will need to approximate the integral in the log likelihood function in (1).
    # There are many ways of doing this, but we will use something called Gauss-Legendre QUADRATURE (Another popular method of approximating the integral is by simulation)
    # We can rewrite the integral in (1) as a (weighted) discrete summation [see assignment text]:
    # where ω_r are the quadrature weights and ξ_r are the quadrature points. ˜γ and σ γ are parameters of the distribution function f(·).
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    # GG: QUADRATURE: a fancy polynomial that can trace out any sort of flexible function (TR layman definition)
    # GG: called quadrature because it uses a quadratic approximation. we can boil down our integral into some small numner of points and then treat is as discrete.
    # GG: variable of integration is ξ_r, so its values are real values on a grid point at which we evaluate the quadrature function (QUADRATURE POINTS/NODES)
    # GG: ω_r are weights, i.e. the density points (QUADRATURE WEIGHTS)
    # GG: f(.) is the normal pdf

    # GG: so in the log likelihood we change the integral with a summation and add the quadrature weights (substituting the "function" of the dγ)

    #---------------------------------------------------
    # a. Before we dive in, let’s learn how to use quadrature. In the folder where this problem set is posted on GitHub, there is a
    # ﬁle called lgwt.jl. This is a function that returns the ω’s and ξ’s for a given choice of K (number of quadrature
    # points) and bounds of integration [a,b]. Let’s practice doing quadrature using the density of the Normal distribution.
    #---------------------------------------------------

    # GG: lgwt() return the ω and the ξ, based on the number of quadrature points and the bounds of integration that you give it

    # define distribution
    d = Normal(0,1) # mean=0, standard deviation=1

    #=
    GG:
    Once we have the distribution deﬁned, we can do things like evaluate its density or probability, take draws from it, etc.
    We want to verify that ∫φ(x)dx is equal to 1 (i.e. integrating over the density of the support equals 1).
    We also want to verify that ∫xφ(x)dx is equal to µ (which for the distribution above is 0).
    When using quadrature, we should try to pick a number of points and bounds that will minimize computational resources,
    but still give us a good approximation to the integral. For Normal distributions, ±4σ will get us there.
    =#

    # GG: name of the game is take bounds of integration which will give us the lowest number of quadrature points but the most coverage of the support of the distribution so that we can have the best of both worlds

    # get quadrature nodes and weights for 7 grid points # GG: typically you want to pick an odd number of grid points; don't know exactly why, maybe it is about symmetry
    nodes, weights = lgwt(7,-4,4) # GG: Input: #points; bounds of integrations
    # GG: output is nodes (ξ) and weights (ω)

    # now compute the integral over the density and verify it's 1
    sum(weights.*pdf.(d,nodes)) # GG: evaluate pdf at nodes and compute weighted sum

    # now compute the expectation and verify it's 0
    sum(weights.*nodes.*pdf.(d,nodes)) # GG: in this case nodes is basically x

    #---------------------------------------------------
    # b. To get some more practice, I’d like you to use quadrature to compute the following integrals:
    #   1. ∫_{−5σ}^{5σ} x^2 f(x)dx where f(·) is the pdf of a N (0,2) distribution using 7 quadrature points # GG: that's the variance of the distribution; should be around 4
    #   2. The same as above, but with 10 quadrature points
    #   The above integrals are the variance of the distribution f. Comment on how well the quadrature approximates the true value.
    #---------------------------------------------------
    d = Normal(0,2)

    nodes, weights = lgwt(7,-10,10)
    sum(weights .* nodes.^2 .* pdf.(d,nodes)) # GG: 1. does not approximate really well

    nodes, weights = lgwt(10,-10,10)
    sum(weights .* nodes.^2 .* pdf.(d,nodes)) # GG: 2. much better approximation

    # it's more accurate with more quadrature points

    # GG: TRADE-OFF: approximation accuracy increases with the number of grid points, but the downside is that it becomes cumbersome
    # GG: you wanna be as parsimonious as possible when you nest it inside an optimization routine

    #---------------------------------------------------
    # c. An alternative to quadrature is MONTE CARLO INTEGRATION. Under this approach, we approximate the integral
    # of f by averaging over a function of many random numbers. Formally, we have that [see assignment text]
    # where D is the number of random draws and where each X i is drawn from a U[a,b] interval:
    #   • With D = 1,000,000, use the formula in (3) to approximate ∫ −5σ x 2 f (x)dx where f (·) is the pdf of a N (0,2) and verify that it is (very) close to 4
    #   • Do the same for ∫ −5σ x f (x)dx where f(·) and verify that it is very close to 0
    #   • Do the same for ∫ −5σ f (x)dx and verify that it is very close to 1
    #   • Comment on how well the simulated integral approximates the true value when D = 1,000 compared to when D = 1,000,000.
    #---------------------------------------------------


    #=
    GG: An alternative to quadrature https://en.wikipedia.org/wiki/Monte_Carlo_integration
    In this case, we approximate the integral by just taking a bunch of random numbers from the
    distribution of the integrand, and then average over those.
    At each draw, we sample points 𝐗_i uniformly on the bounds of integration [a,b]:
    Then we evaluate f(·) at those points and compute an average (corrected by the width of the bounds / # draws)
    [see assignment text] to undertand better
    This can be generalized to higher-order integrals, in which case draws are taken from a joint uniform distribution
    D needs to be large to get a nice approximation
    =#

    b = 10
    a = -10
    c = Uniform(a,b)
    draws = rand(c,1_000_000) # GG: taking a million draws from a uniform distribution with bounds -10 and 10

    # same distribution as in part (b) above "d"

    # estimate of variance # GG: |> is the equivalent of piping in Julia
    (b-a)*mean(draws.^2 .* pdf.(d,draws)) |> println # GG: returns 4

    # estimate of mean
    (b-a)*mean(draws .* pdf.(d,draws)) |> println # GG: returns 0

    # estimate of integral over support
    (b-a)*mean(pdf.(d,draws)) |> println # GG: this is the Monte Carlo integration of equation (3) in the PS

    # now repeat with only 1_000 draws
    draw2 = rand(c,1_000) # GG: this time just a thousand draws
    (b-a)*mean(draw2.^2 .* pdf.(d,draw2)) |> println
    (b-a)*mean(draw2 .* pdf.(d,draw2)) |> println
    (b-a)*mean(pdf.(d,draw2)) |> println

    # GG: definitely not as good with only a thousand draws (he says really really bad)

    #---------------------------------------------------
    # d. Note the similarity between quadrature and Monte Carlo. With quadrature, we approximate the integral with [see assignment text]
    # where ω_i is the quadrature weight and ξ_i the quadrature node. With Monte Carlo, we approximate the integral with [see assignment text]
    # So the “quadrature weight” in Monte Carlo integration is the same (b−a)/D at each node, and the “quadrature node” is a U[a,b] random number
    #---------------------------------------------------
    # Similarity noted
    #=
    GG: Go see text of problem set
    WEIGHTS: Quadrature -> ω_i [changes at each node/point] | Monte Carlo integration -> (b-a)/D [same at each node]
    POINTS: Quadrature -> ξ_i [chosen deterministically by quadrature function] | Monte Carlo integration -> 𝑋_i [drawn randomly]
    The reason you need to take so many draws with MCI is that you are giving all of them the same weight

    GG: quadrature smarter [but outcome of a chosen deterministic (quadratic-like) function], MCI brute force.
    Sometimes Quadrature can't handle integrals, especially crazy integrals (e.g. 50 dimensions)
    There is also "cubiture" that uses a cubic-like function
    =#

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 4
    # Try to modify (but not run!) your code from Question 1 to OPTIMIZE the likelihood function in (2)
    # (This took about 4 hours to estimate on my machine, given the multinomial logit starting values.
    # “Don’t try this at home,” as they say.)
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    # GG: Now using QUADRATURE to approximate the integral in the log likelihood function

    function mixlogit_quad_with_Z(theta, X, Z, y, R) # GG: here R is the number of quadrature points; so that it can be fine-tuned

        alpha = theta[1:end-2]
        gamma = theta[end-1]
        # GG: one additional parameter here σ; (γ is the μ of the gamma distribution; σ is the SD of the gamma distribution)
        sigma = exp(theta[end]) # makes sure sigma is never 0 or negative (which would make the Normal density undefined)
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

        # now write P as a function of variable of integration
        T = promote_type(eltype(X),eltype(theta))
        # function like_int() integrates the likelihood for the given value of parameter vector theta
        function like_int(T,N,J,X,Z,bigAlpha,gamma,sigma,R) # GG: typo: the bounds are actually also function of μ (i.e. γ) so γ should also be an argument
            nodes, weights = lgwt(R,-4*sigma,4*sigma) # GG: as sigma is part of the optimization, nodes and weights will change at every step!
            out = zeros(T,N)
            for r=1:R # GG: for every grid point of our quadrature...
                num   = zeros(T,N,J)
                dem   = zeros(T,N)
                P     = zeros(T,N,J)
                for j=1:J
                    num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*nodes[r])
                    dem .+= num[:,j]
                end
                P = num./repeat(dem,1,J)
                # GG: ... here's where we add the result of the quadrature [see formula of the original integral]
                out .+= vec( weights[r]*( prod(P.^bigY; dims=2) )*pdf(Normal(gamma,sigma),nodes[r]) ) # why do we have to use vec here? because vectors in Julia are "flat" and the vec operator flattens a one-dimensional array.
            end
            return out
        end

        intlike = like_int(T,N,J,X,Z,bigAlpha,gamma,sigma,R) # GG: getting the approximation through the quadrature
        loglike = -sum(log.(intlike)) # GG: evaluating the loglikelihood # GG: it is interesting to notice that sum() will sum over both dimensions N and T because we are using a stacker person-time vector
        return loglike
    end
    startvals = [ .0403744; .2439942; -1.57132; .0433254; .1468556; -2.959103; .1020574; .7473086; -4.12005; .0375628; .6884899; -3.65577; .0204543; -.3584007; -4.376929; .1074636; -.5263738; -6.199197; .1168824; -.2870554; -5.322248; 1.307477; 2]
    startvals = [-.0493358;-.7359427;1.161792;-.0127601;.1499798;-2.438863;.0979789;.6508201;-4.049595;.055819;.2773671;-4.34042;.0203963;-.6880248;-4.794188;.1288023;-.9974547;-7.914539;.118583;-.7813687;-5.609388;4.50982;2.2925]
    td = TwiceDifferentiable(theta -> mixlogit_quad_with_Z(theta, X, Z, y, 7), startvals; autodiff = :forward) # GG: same old stuff; here we specify 7 quadrature points
    # run the optimizer
    theta_hat_mix_optim_ad = optimize(td, startvals, LBFGS(linesearch = BackTracking()), Optim.Options(g_tol = 1e-5, iterations=2, show_trace=true, show_every=1))
    # theta_hat_mix_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mix_mle_ad = theta_hat_mix_optim_ad.minimizer
    println(theta_hat_mix_mle_ad)
    # # evaluate the Hessian at the estimates
    # H  = Optim.hessian!(td, theta_hat_mix_optim_ad) # it bugs out at this point, and I don't now why # GG: he had some problem here
    # # theta_hat_mix_mle_ad_se = sqrt.(diag(inv(H)))
    # # println([theta_hat_mix_mle_ad theta_hat_mix_mle_ad_se]) # these standard errors match Stata

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 5
    # Try to modify (but not run!) your code from Question 1 to optimize the likelihood function in (1),
    # where the integral is approximated by Monte Carlo. Your program will take basically the same form as under
    # Quadrature, but the weights will be slightly different.
    # (This takes even longer because, instead of 7 quadrature points, we need to use many, many simulation draws.)
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    # GG: Now using MONTE CARLO INTEGRATION to approximate the integral in the log likelihood function
    # GG: We can use most of the previous exercise code; we're just gonna change the weights and the dimensionality of the number of points (we need many many more draws with MCI)

    function mixlogit_MC_with_Z(theta, X, Z, y, R)

        alpha = theta[1:end-2]
        gamma = theta[end-1]
        sigma = exp(theta[end]) # makes sure sigma is never 0 or negative (which would make the Normal density undefined)
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]

        # now write P as a function of variable of integration
        T = promote_type(eltype(X),eltype(theta))
        function like_int(T,N,J,X,Z,bigAlpha,gamma,sigma,R)
            draws = 8*sigma.*Base.rand(R).-4*sigma # GG: same problem as above -> should also be function of gamma
            out = zeros(T,N)
            for r=1:R
                num   = zeros(T,N,J)
                dem   = zeros(T,N)
                P     = zeros(T,N,J)
                for j=1:J
                    num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*draws[r])
                    dem .+= num[:,j]
                end
                P = num./repeat(dem,1,J)
                out .+= vec( (8*sigma/R)*( prod(P.^bigY; dims=2) )*pdf(Normal(gamma,sigma),draws[r]) ) # why do we have to use vec here? because vectors in Julia are "flat" and the vec operator flattens a one-dimensional array.
            end
            return out
        end

        intlike = like_int(T,N,J,X,Z,bigAlpha,gamma,sigma,R)
        loglike = -sum(log.(intlike))
        return loglike
    end
    startvals = [ .0403744; .2439942; -1.57132; .0433254; .1468556; -2.959103; .1020574; .7473086; -4.12005; .0375628; .6884899; -3.65577; .0204543; -.3584007; -4.376929; .1074636; -.5263738; -6.199197; .1168824; -.2870554; -5.322248; 1.307477; 2]
    startvals = [-.0493358;-.7359427;1.161792;-.0127601;.1499798;-2.438863;.0979789;.6508201;-4.049595;.055819;.2773671;-4.34042;.0203963;-.6880248;-4.794188;.1288023;-.9974547;-7.914539;.118583;-.7813687;-5.609388;4.50982;2.2925]

    td = TwiceDifferentiable(theta -> mixlogit_MC_with_Z(theta, X, Z, y, 100), startvals; autodiff = :forward)
    # run the optimizer
    theta_hat_mix_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=2, show_trace=true, show_every=1))
    # theta_hat_mix_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mix_mle_ad = theta_hat_mix_optim_ad.minimizer
    println(theta_hat_mix_mle_ad)
    # evaluate the Hessian at the estimates
    # H  = Optim.hessian!(td, theta_hat_mix_optim_ad) # it bugs out at this point, and I don't now why (it's possibly due to me making sigma piecewise above)
    # theta_hat_mix_mle_ad_se = sqrt.(diag(inv(H)))
    # println([theta_hat_mix_mle_ad theta_hat_mix_mle_ad_se]) # these standard errors match Stata

    return nothing
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 6
# Wrap all of your code above into a function and then call that function at the very bottom of your script.
# Make sure you add println() statements after obtaining each set of estimates so that you can read them.
#:::::::::::::::::::::::::::::::::::::::::::::::::::
allwrap()
