#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
# Problem set 3 solutions
# Written LIVE by Giuseppe Grasso over correction session
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
cd("/Users/peppegrass/Documents/GitHub/fall-2020/ProblemSets/PS3-gev/") # GG: sets working directory
pwd() ## GG: prints working directory
readdir() # GG: equivalent to -ls- to see elements of working directory

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1
# Estimate a multinomial logit (with alternative-speciﬁc covariates Z) on the following data set:
#:::::::::::::::::::::::::::::::::::::::::::::::::::
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation


#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #

function allwrap()
#=
GG: Indent and put here
=#
    return nothing
end

# GG: Rather then getting the Hessian, we could also use Bootstrap to get the standard errors (but it'll take a lot longer)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4
# Wrap all of your code above into a function and then call that function at the very bottom of your script.
# Make sure you add println() statements after obtaining each set of estimates so that you can read them.
#:::::::::::::::::::::::::::::::::::::::::::::::::::
allwrap()
