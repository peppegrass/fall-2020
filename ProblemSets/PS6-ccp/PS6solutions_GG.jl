#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
# Problem set 6 solutions
# Written by Tyler Ransom
# Commented by Giuseppe Grasso
#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#

using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM
cd("/Users/peppegrass/Documents/GitHub/fall-2020/ProblemSets/PS6-ccp/") # GG: sets working directory
pwd() ## GG: prints working directory
readdir() # GG: equivalent to -ls- to see elements of working directory

# read in function to create state transitions for dynamic model
include("create_grids.jl")

function allwrap()

    #=
    GG: In this problem set, we will repeat the estimation of the simpliﬁed version of the Rust (1987, Econometrica) bus engine replacement
    model. Rather than solve the model by backwards recursion, we will exploit the renewal property of the replacement decision and estimate
    the model using conditional choice probabilities (CCPs).
    =#

# GG: I - PRELIMINARIES: Loading data, reshaping them, estimating flexible logit model

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1: Follow the directions from PS5 to read in the data (the second CSV ﬁle you read in as part of PS5)
    # and reshape to “long” panel format, calling your long dataset df_long.
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body)

    # create bus id variable
    df = @transform(df, bus_id = 1:size(df,1))

    #---------------------------------------------------
    # reshape from wide to long (must do this thrice be-
    # cause DataFrames.stack() requires doing it one
    # variable at a time)
    #---------------------------------------------------
    # first reshape the decision variable
    dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded,:Zst)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded,:Zst]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfy_long, Not(:variable))

    # next reshape the odometer variable
    dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfx_long, Not(:variable))

    # next reshape the odometer variable
    dfxs = @select(df, :bus_id,:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20)
    dfxs_long = DataFrames.stack(dfxs, Not([:bus_id]))
    rename!(dfxs_long, :value => :Xst)
    dfxs_long = @transform(dfxs_long, time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfxs_long, Not(:variable))

    # join reshaped df's back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    df_long = leftjoin(df_long, dfxs_long, on = [:bus_id,:time])
    sort!(df_long,[:bus_id,:time])

    # variables in array form
    Y      = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
    X      = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
    B      = Vector(df[:,:Branded])
    Xstate = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    Zstate = Vector(df[:,:Zst])
    N      = size(Zstate,1)
    T      = size(Xstate,2)

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2: Estimate a ﬂexible logit model where the dependent variable is the replacement decision and the right
    # hand side is a fully interacted set of the following variables:
    # • Mileage
    # • Mileage^2
    # • Route Usage
    # • Route Usage^2
    # • Branded
    # • Time period
    # • Time period^2
    # Hint: “Fully interacted” means that all terms from 1st order to 7th order (e.g. Odometer^2 × RouteUsage^2 × Branded × time^2).
    # Hint: Julia’s GLM package allows you to easily accomplish this by specifying the interacted variables with asterisks in between them,
    # e.g. Odometer * RouteUsage estimates a model that includes Odometer, Route Usage and the product of the two.
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    θ̂_flex = glm(@formula(Y ~ Odometer * Odometer * RouteUsage *RouteUsage * Branded * time * time), df_long, Binomial(), LogitLink())
    println(θ̂_flex)
    # GG: Using asterisks in GLM package in Julia it gives you the fully interacted model
    # GG: we do this to get probabilities non-parametrically

#= GG: II - DYNAMIC ESTIMATION with CCPs
We will use the ﬂexible logit parameters to generate CCPs which we can use to compute the future value term as alternative to the
backwards recursion we did in PS5.
Recall the model from PS5, where the differenced conditional value function for running the bus (relative to replacing it) was:

(1): $ v_{1t}(x_{t},b) - v_{0t}(x_{t},b) =
        θ_{0} + θ_{1}x_{1t} + θ_{2}b                    [GG: same as static so far, flow utility]
        + β ∫ V_{t+1}(x_{t+1}, b) dF(x_{t+1} | x_{t})     [GG: recursive component]
        $

and where V_{t+1} is the value function and the integral is over transitions in the mileage states x_{t}.
By exploiting the RENEWAL PROPERTY of the decision property, we can express V_{t+1} instead as v_{0t+1}−log(p_{0t+1}).
And since v_{0t+1} corresponds to the renewal action, we know that it is equivalent to [GG: something missing in assignment text; see lecture notes].
Thus, our value function formulation can be simpliﬁed to:

(2): $ v_{1t}(x_{t},b) - v_{0t}(x_{t},b) =
        θ_{0} + θ_{1}x_{1t} + θ_{2}b
        + β ∫ log(p_{0t+1})(x_{t+1}, b) dF(x_{t+1} | x_{t})     [GG: log(p_{0t+1}) instead of V_{t+1}]
        $

and by discretizing the integral, we can simplify this even further to be:

(3): $ v_{1t}(x_{t},b) - v_{0t}(x_{t},b) =
        θ_{0} + θ_{1}x_{1t} + θ_{2}b
        + β ∑_{x_{1,t+1}} log(p_{0t+1})(x_{t+1}, b) [ f_1( x_{1,t+1} | x_{1,t},x_2 ) - f_0( x_{1,t+1} | x_{1,t},x_2 ) ]
        dF(x_{t+1} | x_{t})     [GG: log(p_{0t+1}) instead of V_{t+1}]
        $

where the f_j’s are deﬁned identically as in PS5 .
=#
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3: Estimate the θ’s using (3) and assuming a discount factor of β = 0.9.
    # I will walk you through speciﬁc steps for how to do this:
    #
    # (a) Construct the state transition matrices using the exact same code as in this step of PS5.
    zval,zbin,xval,xbin,xtran = create_grids()
    df_state = DataFrame(Odometer = kron(ones(zbin),xval), RouteUsage = kron(zval,ones(xbin)), time = zeros(size(xtran,1)), Branded = zeros(size(xtran,1)))

    # (b) Compute the future value terms for all possible states of the model. Basically, what we want is −log p 0t+1 evaluated at every possible state of the model (t,b,x 1,t ,x 2 ). The easiest way to do this is to adjust the data that we feed into a predict()
    # function using the ﬂexible logit coefﬁcients from question number 2.
    #   • First, create a data frame that has four variables:
    #       – Odometer reading (equals kron(ones(zbin),xval))
    #       – Route usage (equals kron(ones(xbin),zval))
    #       – Branded (equals 0s of the same size as Odometer and Route usage)
    #       – time (equals 0s of the same size as Branded)
    #   • Now write a function that reads in this data frame, the ﬂexible logit estimates, and the other state variables
    #     (Xstate, Zstate, xtran, etc.)
    #   • Initialize the future value array, which should be a 3-dimensional array of zeros.
    #     The size of the ﬁrst dimension should be the total number of grid points (i.e. the number of rows of xtran).
    #     The second dimension should be 2, which is the possible outcomes of :Branded. The third dimension should be T+1.
    #     Note that the number of rows of the future value array should equal the number of rows of the state data frame.
    #   • Now write two nested for loops:
    #       – Loop over t from 2 to T
    #       – Loop over the two possible brand states { 0,1 }
    #   • Inside all of the for loops, make the following calculations
    #       – Update your state data frame so that the :time variable takes on the value of t and the :Branded variable
    #         takes on the value of b
    #       – Compute p0 using the predict() function applied to your updated data frame and the ﬂexible logit estimates
    #       – Store in the FV array the value of −β log p_0 . Remember that every row of the data frame corresponds to the rows in the
    #         state transition matrix, so you can vectorize this calculation.
    #   • Now multiply the state transitions by the future value term. This requires writing another for loop that goes over the rows in
    #     the original data frame (the one that you read in at the very beginning of this problem set). In other words, loop over i and t.
    #     To get the actual rows of the state transition matrix (since we don’t need to use all possible rows), you should re-use the similar
    #     code from PS5; something like this: FVT1[i,t] = (xtran[row1,:].-xtran[row0,:])'* FV[row0:row0+xbin-1,B[i]+1,t+1]
    #     The purpose of this loop is to map the CCPs from the each-possible-state-is-a-row data frame to the actual data frame we used
    #     to estimate the ﬂexible logit in question 2.
    #   • Your function should return FVT1 in “long panel” format. I used FVT1’[:] to make this conversion, but you should double check that
    #     your i and t indexing of your original data frame matches.

    # GG: Here we're computing FV using CCPs, there's no recursion
    function fvdata(b1,β,df,xbin,zbin,Zstate,Xstate,xtran,N,T,B)
        FV1  = zeros(xbin*zbin,2,T+1)
        FVT1 = zeros(N,T)
        for t=2:T
            for s=0:1
                @with(df, :time    .= t)
                @with(df, :Branded .= s)
                p0 = 1 .- convert(Array{Float64},predict(b1,df)) # GG: the b1 inside predict() is where we'll feed the θ̂_flex we estimated
                FV1[:,s+1,t] = -β.*log.(p0)
            end
        end

        for i=1:N # GG: this is exactly as in PS5, where we multiply the state transitions by the future value term.
            row0 = (Zstate[i]-1)*xbin+1
            for t=1:T
                row1      = row0+Xstate[i,t]-1
                FVT1[i,t] = (xtran[row1,:].-xtran[row0,:])⋅FV1[row0:row0+xbin-1,B[i]+1,t+1]
            end
        end
        return FVT1'[:]
    end
    # GG: calling the function, passing the arguments to it
    @time fvt1 = fvdata(θ̂_flex,0.9,df_state,xbin,zbin,Zstate,Xstate,xtran,N,T,df.Branded)
    df_long = @transform(df_long, fv = fvt1)

    # (c) Estimate the structural parameters.
    #   • Add the output of your future value function as a new column in the original “long panel” data frame.
    #     The easiest way to do this is df long = @transform(df long, fv = fvt1)
    #   • Now use the GLM package to estimate the structural model. Make use of the “offset” function to add the future value term as another
    #     regressor whose coefﬁcient is restricted to be 1. That is: theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink(), offset=df_long.fv)
    θ̂_ccp_glm = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink(), offset=df_long.fv)
    println(θ̂_ccp_glm)

    # GG: if you're dealing with a multinomial model, you might need to not use GLM and code it up yourself

    # (d) Optionally, you can write your own function to estimate a binary logit where you restrict the offset term to have a coefﬁcient of 1.
    # (I will include this code in my solutions.)
    # # You can also do this via Optim, and you'll get the exact same answer
    # FV = reshape(df_long.fv,T,N)'
    # @views @inbounds function likebus_ccp(α,Y,X,B,FV,N,T)
        # like = 0
        # for i=1:N
            # for t=1:T
                # v1    = α[1] + α[2]*X[i,t] + α[3]*B[i] + FV[i,t]
                # dem   = 1 + exp(v1)
                # like -= ( (Y[i,t]==1)*v1 ) - log(dem) # negative of the binary logit likelihood
            # end
        # end
        # return like
    # end
    # θ_true = [2;-.15;1]
    # # evaluate objective function
    # @time likebus_ccp(θ_true,Y,X,B,FV,N,T)
    # @time likebus_ccp(θ_true,Y,X,B,FV,N,T)
    # # estimate likelihood function
    # θ̂_optim = optimize(a -> likebus_ccp(a,Y,X,B,FV,N,T), θ_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
    # θ̂_ccp = θ̂_optim.minimizer
    # println(θ̂_ccp)

    # (e) Wrap all of your code in an empty function as you’ve done with other problem sets. Prepend your wrapper function call (at the very
    # end of the script) with @time so that you can time how long everything takes. (On my machine, everything took under 20 seconds.)
    return nothing
end
@time allwrap()
    # (f) Glory in the power of CCPs!
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
