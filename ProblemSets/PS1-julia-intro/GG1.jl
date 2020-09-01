############# Problem Set 1 #############
############# Giuseppe Grasso, Econ PhD student at LISER/University of Luxembourg (https://liser.elsevierpure.com/en/persons/giuseppe-grasso)
println("Hello, this is my first Julia script!")

############# 1. Initializing variables and practice with basic matrix operations

#1.a
using Random
Random.seed!(1234)
using Distributions
A=rand(Uniform(-5,10),10,7)
B=rand(Normal(-2,15),10,7)
using LinearAlgebra
C=hcat(A[1:5,1:5],B[1:5,6:7])
#currently giving up on D

#1.b
length(A)

#1.c skipped

#1.d
reshape(B,10,7)==
# too late to start this assignment. will work on it later at some point
