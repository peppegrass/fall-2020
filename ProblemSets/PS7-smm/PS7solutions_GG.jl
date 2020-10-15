# problem set 7 solutions

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

function allwrap()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 1 - GMM for simple regression model
    # data & model are same as PS2, question 2
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = log.(df.wage)

    function ols_gmm(β, X, y)
        K = size(X,2)
        N = size(y,1)
        ŷ = X*β
        g = y .- ŷ
        J = g'*I*g
        return J
    end

    # we can also estimate the RMSE of the regression by adding another moment (and assuming homoskedasticity)
    function ols_gmm_with_σ(θ, X, y)
        K = size(X,2)
        N = size(y,1)
        β = θ[1:end-1]
        σ = θ[end]
        g = y .- X*β
        g = vcat(g,( (N-1)/(N-K) )*var(g) .- σ^2) # this builds on the assumption that SSR/(N-K) = σ² 
        J = g'*I*g
        return J
    end

    β_hat_gmm = optimize(b -> ols_gmm(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(β_hat_gmm.minimizer)
    println("ans: ",X\y)
    β_hat_gmm = optimize(b -> ols_gmm_with_σ(b, X, y), rand(size(X,2)+1), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(β_hat_gmm.minimizer)
    println("ans: ",vcat(X\y,sqrt( sum( (y .- X*β_hat_gmm.minimizer[1:end-1]).^2 )/( size(y,1)-size(X,2) ) )))

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 2 - GMM for multinomial logit model
    # data & model are same as PS2, question 5
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
    svals[:,1] = coef(lm(@formula(occ1 ~ age + white + collgrad), df))
    svals[:,2] = coef(lm(@formula(occ2 ~ age + white + collgrad), df))
    svals[:,3] = coef(lm(@formula(occ3 ~ age + white + collgrad), df))
    svals[:,4] = coef(lm(@formula(occ4 ~ age + white + collgrad), df))
    svals[:,5] = coef(lm(@formula(occ5 ~ age + white + collgrad), df))
    svals[:,6] = coef(lm(@formula(occ6 ~ age + white + collgrad), df))
    svals[:,7] = coef(lm(@formula(occ7 ~ age + white + collgrad), df))
    svals = svals[:,1:6] .- svals[:,7]
    svals = svals[:]
    
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
        
        loglike = -sum( bigY.*log.(P) )
        
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
                g[(j-1)*K+k] = mean( (bigY[:,j] .- P[:,j]) .* X[:,k] )
            end
        end
        J = N*g⋅g
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
        
        P = exp.(X*bigα) ./ sum.(eachrow(exp.(X*bigα)))

        g = bigY[:] .- P[:]

        J = g'*I*g
        return J
    end
    #α_zero = zeros(6*size(X,2))
    #α_rand = rand(6*size(X,2))
    α_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
    #α_start = α_true.*rand(size(α_true))
    #println("starting values: ",α_rand)
    
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

    compare = DataFrame(mle = α_hat_mle, gmm = α_hat_gmm, gmm_ad = α_hat_gmm_ad, gmm_ad_newton = α_hat_gmm_ad_newton, gmm_ad_fr_bad = α_hat_gmm_ad_fr_bad)
    println(compare)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3 - simulate data from a logit model
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    function sim_logit(N=100_000,J=4) # putting "=" after the arguments here sets default values for these arguments
        # generate X, β and P
        X = hcat(ones(N),randn(N),2 .+ 2 .* randn(N))
        if J==4
            β = hcat([1, -1, 0.5],[-2, 0.5, 0.3],[0, -0.5, 2],zeros(3))
        else
            β = 2 .* rand(size(X,2),J) .- 1
        end
        P = exp.(X*β) ./ sum.(eachrow(exp.(X*β)))
        # draw choices
        draw = rand(N)
        Y = zeros(N)
        for j=1:J
            Ytemp = sum(P[:,j:J];dims=2) .> draw
            Y += Ytemp
        end
        return Y,X
    end
    ySim,XSim = sim_logit()
    α_hat_sim_optim = optimize(a -> mlogit_mle(a, XSim, ySim), rand(9), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    α_hat_sim_mle = α_hat_sim_optim.minimizer
    println("simulated mlogit MLE estimates: ",α_hat_sim_mle)

    # now with drawing from Gumbel instead
    function sim_logit_w_gumbel(N=100_000,J=4) # putting = here sets default values for these arguments
        # generate X and β (No P needed)
        X = hcat(ones(N),randn(N),2 .+ 2 .* randn(N))
        if J==4
            β = hcat([1, -1, 0.5],[-2, 0.5, 0.3],[0, -0.5, 2],zeros(3))
        else
            β = 2 .* rand(size(X,2),J) .- 1
        end
        # draw choices
        ϵ = rand(Gumbel(0,1),N,J)
        Y = argmax.(eachrow(X*β .+ ϵ))
        return Y,X
    end
    ySim,XSim = sim_logit_w_gumbel()
    α_hat_sim_optim = optimize(a -> mlogit_mle(a, XSim, ySim), rand(9), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    α_hat_sim_mle = α_hat_sim_optim.minimizer
    println("simulated mlogit MLE estimates: ",α_hat_sim_mle)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 4 - run SMM.parallelNormal()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    MA = SMM.parallelNormal()
    dc = SMM.history(MA.chains[1])
    dc = dc[dc[:accepted].==true, :]
    println(describe(dc))

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 5 - estimate mlogit by SMM
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    X = [ones(size(df,1),1) df.age df.white df.collgrad]
    y = df.occupation
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
    #end
    function mlogit_smm_overid(α, X, y, D)
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        bigỸ = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigα = [reshape(α,K,J-1) zeros(K)]

        Random.seed!(1234)
        for d=1:D
            # draw choices
            ε = rand(Gumbel(0,1),N,J)
            ỹ = argmax.(eachrow(X*bigα .+ ε))
            for j=1:J
                bigỸ[:,j] .+= (ỹ.==j)*(1/D)
            end
        end

        g = bigY[:] .- bigỸ[:]

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
# question 6
#:::::::::::::::::::::::::::::::::::::::::::::::::::
@time allwrap()

