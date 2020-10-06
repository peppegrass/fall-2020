#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
# Problem set 5 solutions
# Written by Tyler Ransom
# Commented by Giuseppe Grasso
# Recording available in Notability
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
cd("/Users/peppegrass/Documents/GitHub/fall-2020/ProblemSets/PS5-ddc/") # GG: sets working directory
pwd() ## GG: prints working directory
readdir() # GG: equivalent to -ls- to see elements of working directory

# read in function to create state transitions for dynamic model
include("create_grids.jl")

function allwrap()

    #=
    GG: In this problem set, we will explore a simpliﬁed version of the Rust (1987, Econometrica) bus engine
    replacement model. Let’s start by reading in the data.
    =#

# GG: I - PRELIMINARIES: Loading data and reshaping them

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1: Reshaping the data
    # Reshape the data into “long” panel format, calling your long dataset df_long.
    # I have included code on how to do this in the PS5starter.jl ﬁle that accompanies this problem set.
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    # load in the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body)

    # create bus id variable
    df = @transform(df, bus_id = 1:size(df,1))

    #---------------------------------------------------
    # reshape from wide to long (must do this twice be-
    # cause DataFrames.stack() requires doing it one
    # variable at a time)
    #---------------------------------------------------
    # first reshape the decision variable
    dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfy_long, Not(:variable))

    # next reshape the odometer variable
    dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfx_long, Not(:variable))

    # join reshaped df's back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    sort!(df_long,[:bus_id,:time])

# GG: II - STATIC ESTIMATION

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2: Estimate a STATIC version of the model
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    #=
    GG: The model we would like to estimate is Harold Zurcher’s decision to run buses in his ﬂeet.
    Zurcher’s FLOW UTILITY OF RUNNING (i.e. not replacing) a bus is

    (1): $u_{1}(x_{1t},b)= θ_{0}+ θ_{1}x_{1t} + θ_{2}b$ [see assignment text]

    where x_{1t} is the mileage on the bus’s odometer (in 10,000s of miles) and b is a dummy variable indicating
    whether the bus is branded (meaning its manufacturer is high-end). The choice set is {0,1} where 0 denotes
    replacing the engine.
    Estimate the θ parameters assuming Zurcher is completely myopic.
    This amounts to estimating a simple binary logit model.
    (Note: you may estimate this any way you wish. I would recommend using the GLM package, but you may also
    use Optim with your own log likelihood function.)
    =#

    θ̂_glm = glm(@formula(Y ~ Odometer + Branded), df_long, Binomial(), LogitLink()) # GG: estimating using GLM package. Pooled binary logit using reshaped long data
    println(θ̂_glm)

#= GG: III - DYNAMIC ESTIMATION
Now I will walk you through how to estimate the DYNAMIC version of this model using BACKWARDS RECURSION.
With discount factor β, the DIFFERENTIATED CONDITIONAL VALUE FUNCTION for running the bus (relative to replacing it) is

(2): $ v_{1t}(x_{t},b) - v_{0t}(x_{t},b) =
        θ_{0} + θ_{1}x_{1t} + θ_{2}b                    [GG: same as static so far, flow utility]
        + β ∫ V_{t+1}(x_{t+1}, b) dF(x_{t+1} | x_{t})     [GG: recursive component]
        $

where V_{t+1} is the VALUE FUNCTION and the integral is over TRANSITIONS in the mileage states x_{t}.

We will APPROXIMATE the INTEGRAL with a SUMMATION, which means that we will specify a DISCRETE MASS FUNCTION for f(x_{t+1}  |x_{t}).
This probability mass function depends on the current odometer reading (x_{1t}),
whether the engine is newly replaced (i.e. d_{t−1}=0),
and on the value of another state variable x_2 which measures the usage intensity of the bus’s route
(i.e. high values of x_2 imply a low usage intensity and vice versa).
We discretize the mileage transitions into 1,250-mile bins (i.e. 0.125 units of x-{1t}).
We specify x_2 as a discrete uniform distribution ranging from 0.25 to 1.25 with 0.01 unit increments.

Formally, we are DISCRETELY (but not discreetly!) APPRXIMATING an EXPONENTIAL DISTRIBUTION:

(3): [see assignment text]

You will not need to program (3); I will provide code for this part. Under this formulation, (2) can be written as:

(4): [see assignment text]

Finally, we can simplify (4) since we know that V_{t+1 = log( ∑_{k} exp(v_{k,t+1}) ) when we assume
that unobserved utility is drawn from a T1EV distribution (as we do here):

(5): [see assignment text]

Estimation of our dynamic model now requires two steps:

#1 SOLVING THE MODEL
First, we need to solve the value functions for a given value of our parameters θ.
The way we do this is by BACKWARDS RECURSION. We know that V_{t+1} = 0 in our ﬁnal period (i.e. when t = T).
Then we work backwards to obtain the future value at every possible state in our model.
This will include many states that do not actually show up in our data.

# 2 ESTIMATING THE MODEL
Second, once we’ve solved the value functions, we use maximum likelihood to estimate the parameters θ.
The log likelihood function in this case is simply:
(6)-(7): [see assignment text]

Now estimate the θ’s assuming that Zurcher discounts the future with discount factor β = 0.9.
I will walk you through speciﬁc steps for how to do this
=#

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3a: Read in the data for the dynamic model.
    # This can be found at the same URL as listed at the top of p. 2, but remove the "Beta0" from the CSV ﬁlename.
    # Rather than reshaping the data to “long” format as in question 1, we want to keep the data in “wide” format.
    # Thus, columns :Y1 through :Y20 should be converted to an array labeled Y which has
    # dimension 1000 × 20 where N = 1000 and T = 20. And similarly for columns starting with :Odo and :Xst.
    # Variables :Xst* and :Zst keep track of which discrete bin of the f j ’s the given observation falls into.
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    # [ Load in the data ]
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body)
    Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
    X = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
    Z = Vector(df[:,:RouteUsage])
    B = Vector(df[:,:Branded])
    N = size(Y,1)
    T = size(Y,2)
    Xstate = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    Zstate = Vector(df[:,:Zst])


    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3b: Generate state transition matrices
    # Construct the state transition matrices, which are the f_j’s in (3). To do so, simply run the following code:
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    zval,zbin,xval,xbin,xtran = create_grids() # GG: creates grid points based on points given. See .jl file pf function
    # GG: returns bin numbers and associated values for z (rout usage) and x (odometer reading); and transition values for the X

    #=
    GG: zval and xval are the grids deﬁned at the bottom of p. 2, which respectively correspond to the route usage
    and odometer reading. zbin and xbin are the number of bins in zval and xval, respectively.
    xtran is a (zbin*xbin)×xbin Markov transition matrix that gives the probability of falling into each x_{1,t+1} bin
    given values of x_{1,t} and x_2 , according to the formula in (3).

    Note: A Markov transition matrix is a matrix where each row sums to 1 and moving from e.g. column 1 to column 4
    within a row gives the probability of moving from state 1 to state 4. Check out the Wikipedia page for more information
    =#

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3c: Compute the future value terms for all possible states of the model
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # • First, initialize the future value array, which should be a 3-dimensional array of zeros.
    #   The size of the ﬁrst dimension should be the total number of grid points (i.e. the number of rows of xtran).
    #   The second dimension should be 2, which is the possible outcomes of :Branded.
    #   The third dimension should be T+1.
    # • Now write four nested for loops over each of the possible states:
    #   - Loop backwards over t from T+1 to 1
    #   – Loop over the two possible brand states {0,1}
    #   – Loop over the possible permanent route usage states (i.e. from 1 to zbin)
    #   – Loop over the possible odometer states (i.e. from 1 to xbin)
    # • Inside all of the for loops, make the following calculations
    #   – Create an object that marks the row of the transition matrix that we need to be looking at
    #     (based on the loop values of the two gridded state variables).
    #     This will be x + (z-1)*xbin (where x indexes the mileage bin and z indexes the route usage bin),
    #     given how the xtran matrix was constructed in the create grids() function.
    #   – Create the conditional value function for driving the bus (v_{1t}) based on the values of the state
    #     variables in the loop (not the values observed in the data). For example, for the mileage (x_{1t}),
    #     you should plug in xval[x] rather than :Odo. The difﬁcult part of the conditional value function is
    #     the discrete summation over the state transitions. For this, you need to grab the appropriate row
    #     (and all columns) of the xtran matrix, and then take the dot product with that and the all possible x_{1t}
    #     rows of the FV matrix. You should end up with something like xtran[row,:]'*FV[(z-1)*xbin+1:z*xbin,b+1,t+1]
    #     where b indexes the branded dummy and t indexes time periods.
    #   – Now create the conditional value function for replacing the engine (v_{0t}). For this, we repeat the same
    #     process as with v_{1t} except the θ’s are NORMALIZED to be 0. The code for the expected future value is
    #     the same as for v_{1t} with the exception that mileage resets to 0 after replacement, so instead of grabbing
    #     xtran[row,:] we want xtran[1+(z-1)*xbin,:].
    #   – Finally, update the future value array in period t by storing β log( exp(v_{0t}) + exp(v—1t}) ) in the t_th
    #     slice of the 3rd dimension of the array. This will be the new future value term for period t−1.
    #     Remember to set β = 0.9
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    @views @inbounds function likebus(α,β,Y,B,N,T,X,Z,Zstate,Xstate,xtran,zbin,xbin,xval)

        # FV in period T+1 = 0 since this is a finite horizon problem
        FV=zeros(zbin*xbin,2,T+1) # GG: initialize future values; initialize the future value array, which should be a 3-dimensional array of zeros.
        # GG: 2 columns is Branded=1 vs Branded=0

        # First loop: solve the backward recursion problem given the values of α
        # This will give the future value for *all* possible states that we might visit
        # This is why the FV array does not have an individual dimension
        for t=T:-1:1  # GG: Loop backwards over t from T+1 to 1; going backwards over time every period
            for b=0:1   # GG: Loop over the two possible brand states {0,1}; based on branded or not
                for z=1:zbin # GG: Loop over the possible permanent route usage states (i.e. from 1 to zbin); which bin am I in the x2 state variable
                    for x=1:xbin # GG: Loop over the possible odometer states (i.e. from 1 to xbin); which hin am I in the X1t state variable
                        # inputs to FV

                        # GG: object that marks the row of the transition matrix; to figurate the transition probabilities
                        row = x + (z-1)*xbin # which row of xtran should we look at? depends on milage bin x and route usage z

                        # GG: conditional value function for driving the bus (v_{1t})

                        v1  = α[1] + α[2]*xval[x] + α[3]*b + xtran[         row,:]⋅FV[(z-1)*xbin+1:z*xbin,b+1,t+1] # mileage bin is x, route usage z is permanent
                        # GG: flow utility component + summation over state transitions
                        # GG: set of probs xtran[row,:] DOT PRODUCT Future Values (FV)

                        # GG: conditional value function for replacing the engine (v_{0t})
                        v0  =                                xtran[1+(z-1)*xbin,:]⋅FV[(z-1)*xbin+1:z*xbin,b+1,t+1] # the engine got replaced => mileage is 0, so first bin

                        # FV is discounted log sum of the exponentiated conditional value functions #GG: Updating Future Value terms!!!
                        FV[row,b+1,t] = β*log(exp(v1) + exp(v0))
                        #GG: that's where the recursion comes in!
                    end
                end
            end
        end

        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        # Question 3d: Construct the log likelihood using the future value terms from the previous step and only
        # using the observed states in the data. This will entail a for loop over buses and time periods.
        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        # • Initialize the log likelihood value to be 0. (We will iteratively add to it as we loop over observations in the data)
        # • Create a variable that indexes the state transition matrix rows for the case where the bus has been replaced.
        #   This will be the same 1+(z-1)*xbin as in the conditional value function v 0t above. However, we need to plug in :Zst from the data rather than a hypothetical value z.
        # • Create a variable that indexes the state transition matrix rows for the case where the bus has not been replaced.
        #   This will be the same x + (z-1)*xbin as in v1t  above, except we substitute :Xst and :Zst for x and z.
        # • Now create the ﬂow utility component of v_{1t} − v_{0t} using the actual observed data on mileage and branding.
        # • Next, we need to add the appropriate discounted future value to round out our calculation of v_{1t} − v_{0t}.
        #   Here, we can difference the f_j’s as in (5). You should get something like (xtran[row1,:].-xtran[row0,:])'*FV[row0:row0+xbin-1,B[i]+1,t+1]
        # • Finally, create the choice probabilities for choosing each option as written in (7) and then create the
        #   log likelihood according to the summation in (6).

        # Second loop: form the likelihood given the future values implied by the previous α guess
        # Here, we will take the state-specifc FV's calculated in the first loop
        # But we will only use in the likelihood those state value that are actually visited
        like = 0 # GG: Initialize the log likelihood value to be 0.
        for i=1:N
            # GG: vector of indices in the ACTUAL DATA (no replacement)
            row0 = (Zstate[i]-1)*xbin+1 # this is the same argument as the index of xtran in v0 above, but we use the actual Z
            #z2   = (Zstate[i]-1)*xbin+1 # this is the same argument as the 1st part of the 1st index of the FV array above, but using actual Z
            #z3   = z2+xbin-1            # this is the same argument as the 2nd part of the 1st index of the FV array above, but using actual Z
            for t=1:T
                # GG: vector of indices in the ACTUAL DATA (replacement)
                row1  = Xstate[i,t] + (Zstate[i]-1)*xbin                                                                # this is the same as row in the first loop, except we use the actual X and Z
                v1    = α[1] + α[2]*X[i,t] + α[3]*B[i] + (xtran[row1,:].-xtran[row0,:])⋅FV[row0:row0+xbin-1,B[i]+1,t+1] # using the same formula as v1 above, except we use observed values of X and B, and we difference the transitions
                dem   = 1 + exp(v1)
                like -= ( (Y[i,t]==1)*v1 ) - log(dem) # negative of the binary logit likelihood
            end
        end
        return like
    end

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3e: Wrap all of the code you wrote in (c) and (d) into a function and set up the function so that it can be
    # passed to Optim. For example, you will need to return the negative of the log likelihood and you will need to have
    # the ﬁrst argument be the θ vector that we are trying to estimate
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    # GG: See row 213

    θ_start = rand(3)
    θ_true  = [2; -.15; 1]
    # how long to evaluate likelihood function once?
    println("Timing (twice) evaluation of the likelihood function")
    @time likebus(θ_start,0.9,Y,B,N,T,X,Z,Zstate,Xstate,xtran,zbin,xbin,xval)
    @time likebus(θ_start,0.9,Y,B,N,T,X,Z,Zstate,Xstate,xtran,zbin,xbin,xval)
    # estimate likelihood function
    θ̂_optim = optimize(a -> likebus(a,0.9,Y,B,N,T,X,Z,Zstate,Xstate,xtran,zbin,xbin,xval), θ_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true))
    θ̂_ddc = θ̂_optim.minimizer
    println(θ̂_ddc)

    return nothing
end

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3f: On the same line as the function, prepend the function declaration with the macros so that your code
    # says @views @inbounds function myfun() rather than function myfun(). This will give you more performant code.
    # On my machine, it cut the computation time in half.
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    # GG: See row 213
    # @inbounds turns off dimensions checkin by Julia; consumes time, if your sure comformability is right you can turn it off to save time
    # @views treats items as immutable, makes things more efficiently
    # GG: You gotta make sure your functions work firt. Then you use these options to make things faster
    # GG: in Matlab it takes about 25mins; in Julia only 5mins; if we used the TwiceDifferentiable object, it'd be even faster


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3g: Wrap all of your code in an empty function as you’ve done with other problem sets
#:::::::::::::::::::::::::::::::::::::::::::::::::::
allwrap()

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3h: Try executing your script to estimate the likelihood function. This took about 4 minutes
# on my machine when I started from the estimates of the static model in Question 2.
#:::::::::::::::::::::::::::::::::::::::::::::::::::

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3i: Pat yourself on the back and grab a beverage of your choice, because that was a lot of work!
#:::::::::::::::::::::::::::::::::::::::::::::::::::
