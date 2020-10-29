# problem set 8 solutions

using Random
using LinearAlgebra
using Statistics
using Distributions
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM
using FreqTables
using ForwardDiff
using LineSearches
using SMM
using MultivariateStats

include("lgwt.jl")

function wrapper()
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 1 - read in data and simple regression
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS8-factor/nlsy.csv"
    df = CSV.read(HTTP.get(url).body)
    est1 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
    println(est1)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 2 - compute correlation of  ASVAB scores
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    asvabs = convert(Array,df[:,end-5:end])
    cors = cor(asvabs)
    cordf = DataFrame(cor1 = cors[:,1], cor2 = cors[:,2], cor3 = cors[:,3], cor4 = cors[:,4], cor5 = cors[:,5], cor6 = cors[:,6]) 
    display(cordf)
    # these are all pretty highly correlated; some more than others

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 3 - add ASVAB scores to regression
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    est2 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
    println(est2)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 4 - PCA of ASVAB scores
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    asvabMat = convert(Array,df[:,end-5:end])'
    M = fit(PCA, asvabMat; maxoutdim=1)
    asvabPCA = MultivariateStats.transform(M, asvabMat)
    df = @transform(df, asvabPCA = asvabPCA[:])
    est3 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df)
    println(est3)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 5 - Factor Analysis of ASVAB scores
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    asvabMat = convert(Array,df[:,end-5:end])'
    M = fit(FactorAnalysis, asvabMat; maxoutdim=1)
    asvabFac = MultivariateStats.transform(M, asvabMat)
    df = @transform(df, asvabFac = asvabFac[:])
    est4 = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFac), df)
    println(est4)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # question 6 - Estimate full factor model by MLE
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    X = [df.black df.hispanic df.female df.schoolt df.gradHS df.grad4yr ones(size(df,1),1)]
    y = df.logwage
    Xfac = [df.black df.hispanic df.female ones(size(df,1),1)]
    asvabs = [df.asvabAR df.asvabCS df.asvabMK df.asvabNO df.asvabPC df.asvabWK]

    # likelihood obj function
    function factor_model(θ, X, Xfac, Meas, y, R)
        K = size(X,2)    # no. covariates in outcome equation
        L = size(Xfac,2) # no. covariates in measurement eqns
        J = size(Meas,2) # no. measurements
        N = length(y)    # no. observations

        # organize the parameter vector
        # first J*L elements are the coefficients of the measurement system
        γ = reshape(θ[1:J*L],L,J)
        # next K elements are the coefficients of the outcome equation
        β = θ[J*L+1:J*L+K]
        # next J+1 elements are the factor loadings
        α = θ[J*L+K+1:J*L+K+J+1]
        # last J+1 elements are the variances (stdevs)
        σ = θ[end-J:end]

        # get grid nodes and weights for integral
        ξ,ω = lgwt(R,-5,5)

        # now build the likelihood
        T = promote_type(eltype(X),eltype(θ))
        like  = zeros(T,N)
        for r=1:R
            Mlike = zeros(T,N,J)
            for j=1:J
                Mres        = Meas[:,j] .- (Xfac*γ[:,j] .+ α[j]*ξ[r])
                sdj         = sqrt(σ[j]^2)
                Mlike[:,j] .= (1 ./ sdj).*pdf.(Normal(0,1),Mres./sdj)
            end
            Yres  = y .- (X*β .+ α[end]*ξ[r])
            sdy   = sqrt(σ[end]^2)
            Ylike = (1 ./ sdy).*pdf.(Normal(0,1),Yres./sdy) 

            like += ω[r].*prod(Mlike; dims=2).*Ylike.*pdf(Normal(0,1),ξ[r])
        end
        loglike = -sum(log.(like))
        return loglike
    end

    # SMM objective function
    function factor_model_smm(θ, X, Xfac, Meas, y, D)
        K = size(X,2)    # no. covariates in outcome equation
        L = size(Xfac,2) # no. covariates in measurement eqns
        J = size(Meas,2) # no. measurements
        N = length(y)    # no. observations

        # organize the parameter vector
        # first J*L elements are the coefficients of the measurement system
        γ = reshape(θ[1:J*L],L,J)
        # next K elements are the coefficients of the outcome equation
        β = θ[J*L+1:J*L+K]
        # next J+1 elements are the factor loadings
        α = θ[J*L+K+1:J*L+K+J+1]
        # last J+1 elements are the variances (stdevs)
        σ = θ[end-J:end]

        # simulate the model given the parameter values
        Random.seed!(1234)
        T = promote_type(eltype(X),eltype(θ))
        M̃ = zeros(T,N,J)
        ỹ = zeros(T,N)
        for d=1:D
            ξ = randn(N)
            for j=1:J
                M̃[:,j] .+= (Xfac*γ[:,j] .+ α[j].*ξ .+ sqrt(σ[j].^2).*randn(N)).*(1/D)
            end
            ỹ .+= (X*β .+ α[end].*ξ .+ sqrt(σ[end].^2).*randn(N)).*(1/D)
        end
        g = vcat(Meas[:],y) .- vcat(M̃[:],ỹ)
        J = g'*I*g
        return J
    end
    svals = vcat(Xfac\asvabs[:,1],
                 Xfac\asvabs[:,2],
                 Xfac\asvabs[:,3],
                 Xfac\asvabs[:,4],
                 Xfac\asvabs[:,5],
                 Xfac\asvabs[:,6],
                 X\y,
                 rand(7),
                 .5*ones(7)
                )
    stata = vcat(-.9343684, -.6783938, -.0167224, .362379, -.4309737, -.4427523, .399414, -.003017, -.8084006, -.6365506, .1950629, .2234858, -.3418862, -.4675659, .2531868, .0513019, -.7595222, -.5958526, .2988241, .1545899, -.9353962, -.750025, .0496955, .3432385, -.1870249, -.0805532, -.1349807, -.0024857, .1064984, .2389176, 2.154472, .7704785, .5507262, .816202, .6431685, .7398663, .6622146, .1051398, sqrt(.2306496),sqrt(.6136927),sqrt(.1903943),sqrt(.5309717),sqrt(.3110141),sqrt(.3743002),sqrt(.2188733))
    # run the optimizer for MLE
    td = TwiceDifferentiable(th -> factor_model(th, X, Xfac, asvabs, y, 9), svals; autodiff = :forward)
    println("optimizing quadrature full factor model")
    θ̂_optim_ad = optimize(td, svals, Newton(linesearch = BackTracking()), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=1))
    θ̂_mle_optim_ad = θ̂_optim_ad.minimizer
    loglikeval = θ̂_optim_ad.minimum
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, θ̂_mle_optim_ad)
    θ̂_mle_optim_ad_se = sqrt.(diag(inv(H)))

    # run the optimizer for MLE from Stata starting values
    td = TwiceDifferentiable(th -> factor_model(th, X, Xfac, asvabs, y, 9), stata; autodiff = :forward)
    println("optimizing quadrature full factor model")
    θ̂_optim_ad_fr_stata = optimize(td, stata, Newton(linesearch = BackTracking()), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=1))
    θ̂_mle_optim_ad_fr_stata = θ̂_optim_ad_fr_stata.minimizer
    loglikeval_fr_stata = θ̂_optim_ad_fr_stata.minimum
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, θ̂_mle_optim_ad)
    θ̂_mle_optim_ad_se_fr_stata = sqrt.(diag(inv(H)))

    # run the optimizer for SMM from Stata starting values
    td = TwiceDifferentiable(th -> factor_model_smm(th, X, Xfac, asvabs, y, 3_000), stata; autodiff = :forward)
    println("optimizing SMM full factor model")
    θ̂_sm_optim_ad = optimize(td, stata, Newton(linesearch = BackTracking()), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=1))
    θ̂_smm_optim_ad = θ̂_sm_optim_ad.minimizer
    errval = θ̂_sm_optim_ad.minimum
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, θ̂_smm_optim_ad)
    θ̂_smm_optim_ad_se = sqrt.(diag(inv(H)))

    # print output
    eqs = vcat(fill("asvabAR",size(Xfac,2),1),
               fill("asvabCS",size(Xfac,2),1),
               fill("asvabMK",size(Xfac,2),1),
               fill("asvabNO",size(Xfac,2),1),
               fill("asvabPC",size(Xfac,2),1),
               fill("asvabWK",size(Xfac,2),1),
               fill("wage",size(X,2),1),
               repeat(["asvabAR", "asvabCS", "asvabMK", "asvabNO", "asvabPC", "asvabWK", "wage"],2,1),
              "overall")
    namer = vcat(repeat(["black","hispanic","female","intercept"],size(asvabs,2),1), 
                 ["black","hispanic","female","yrs school","gradHS","grad4yr","intercept"],
                 fill("loading",size(asvabs,2)+1,1),
                 fill("stdev",size(asvabs,2)+1,1),
                "loglike") 
    results = DataFrame(equation = vec(eqs), variable = vec(namer), coef_mle = vcat(vec(θ̂_mle_optim_ad),loglikeval), se_mle = vcat(vec(θ̂_mle_optim_ad_se),missing), coef_mle_fr_stata = vcat(vec(θ̂_mle_optim_ad_fr_stata),loglikeval_fr_stata), se_mle_fr_stata = vcat(vec(θ̂_mle_optim_ad_se_fr_stata),missing), coef_smm = vcat(vec(θ̂_smm_optim_ad),missing), se_smm = vcat(vec(θ̂_smm_optim_ad_se),missing),coef_stata = vcat(vec(stata),17524.022))
    println(results)

    return nothing
end
wrapper()
