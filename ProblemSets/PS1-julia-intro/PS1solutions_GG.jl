#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#
# Problem set 1 solutions
# Written by Tyler Ransom
# Commented by Giuseppe Grasso
#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~=#

using DataFrames
using FreqTables
using CSV
using JLD
using Random
using LinearAlgebra
using Statistics
using Distributions
cd("/Users/peppegrass/Documents/GitHub/fall-2020/ProblemSets/PS1-julia-intro/") # GG: sets working directory
pwd() ## GG: prints working directory
readdir() # GG: equivalent to -ls- to see elements of working directory
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1
# Initializing variables and practice with basic matrix operations
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# a. Set the seed
Random.seed!(1234)
#=
GG: It seems to me that Random.seed!() only applies to the next command.
Hence the use of function q1() so that it applies to everything.
But I'm not sure about it.
=#

#=
GG: The content of the function cannot be called from the .jl
You either copy and paste it in the REPL or call the functions (as done in the end of this script)
=#
function q1()
    ## a. i.-iv. create matrices

    # A 10×7- random numbers distributed U [−5,10]
    A = 15*rand(10,7).-5 # GG: same result as using Distributions A=rand(Uniform(-5,10),10,7). Not sure why you have 15 instead of 10 here, but it works.

    # B 10×7- random numbers distributed N (−2,15) [st dev is 15]
    B = 15*randn(10,7).-2 # GG: same result as using Distributions B=rand(Normal(-2,15),10,7)

    # C 5×7- the ﬁrst 5 rows and ﬁrst 5 columns of A and the last two columns and ﬁrst 5 rows of B
    C = cat(A[1:5,1:5],B[1:5,end-1:end]; dims=2) # GG: notice end arguments

    # D 10×7- where D i,j= A i,jif A i,j≤ 0, or 0 otherwise
    D = A.*(A.<=0) # GG: D contains only negative elements of A; rest filled with 0s
    # GG: very elegant subsetting; had no idea how to do it

    ## b. Use a built-in Julia function to list the number of elements of A
    println(length(A))

    ## c. Use a series of built-in Julia functions to list the number of unique elements of D
    println(length(unique(A)))

    ## d. Using the reshape() function, create a new matrix called E which is the ‘vec’ operator applied to B.
    ## Can you ﬁnd an easier way to accomplish this?

    # one way:
    E = reshape(B,length(B)) # GG: piles columns of B one above the other
    # easy way:
    E = B[:] ## GG: simple as that

    ## e. Create a new array called F which is 3-dimensional and contains
    ## A in the ﬁrst column of the third dimension and
    ## B in the second column of the third dimension
    F = cat(A,B; dims=3) # GG: making a cuboid (3-dimensional tensor) out of matrices A and B; see https://improbable-research.github.io/keanu/docs/tensors/

    ## f. Use the permutedims() function to twist F so that it is now F 2×10×7 instead of F 10×7×2 . Save this new matrix as F.
    F = permutedims(F, [3 1 2]) # GG: now F has 7 third dimensions, each containing a 2x10 matrix

    ## g. Create a matrix G which is equal to B ⊗ C (the Kronecker product of B and C). What happens when you try C ⊗ F?
    G = kron(B,C) # GG: A (5x10=50)x(7x7=49) matrix; see https://en.wikipedia.org/wiki/Kronecker_product
    #kron(C,F) # it does work; not sure whether it is that tensor of different dimension cannot be kronecher-multiplied (don't think so) or that this Julia command isn't suited for such operation

    ## h. Save the matrices A, B, C, D, E, F and G as a .jld ﬁle named matrixpractice.
    save("matrixpractice.jld","A",A,"B",B,"C",C,"D",D,"E",E,"F",F,"G",G)
        #=
        GG: "matrixpractice.jld" can be read back with
        d = load("matrixpractice.jld")
        where d is a dictionary
        Other examples of .jld loading are provided in Exercise 4
        =#

    ## i. Save only the matrices A, B, C, and D as a .jld ﬁle called firstmatrix.
    save("firstmatrix.jld","A",A,"B",B,"C",C,"D",D)

    ## j. Export C as a .csv ﬁle called Cmatrix. You will ﬁrst need to transform C into a DataFrame.
    CSV.write("Cmatrix.csv",DataFrame(C))

    ## k. Export D as a tab-delimited .dat ﬁle called Dmatrix. You will ﬁrst need to transform D into a DataFrame.
    CSV.write("Dmatrix.dat",DataFrame(D); delim="\t")
    return A,B,C,D
end
#=
l. Wrap a function deﬁnition around all of the code for question 1. Call the function q1().
The function should have 0 inputs and should output the arrays A, B, C and D.
At the very bottom of your script you should add the code A,B,C,D = q1().
=#


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2
# Practice with loops and comprehensions
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function q2(A,B,C)
    ## a. Write a loop or use a comprehension that computes the element-by-element product of A and B. Name the new matrix AB.
    ## Create a matrix called AB2 that accomplishes this task without a loop or comprehension.

    #=
    GG: Using a COMPREHENSION to compute the element-by-element product of A and B; which has same dimension 10x7
    What is a comprehension? It allows the user in 1 line of code to create an object that could be a complex formula.
    e.g. computing a present value by PV = sum([β^t*Y[t] for t=1:T])
    or a shift-share instrument for what matters
    =#
    AB = [A[i,j]*B[i,j] for i=1:size(A,1),j=1:size(A,2)] # GG: could have been size(B,1) and size(B,2); what matter are the 1 and 2 to indicate dimensions

    AB2 = A.*B ## GG: doing it with the ELEMENT-BY-ELEMENT multiplication oparator .*

    ## b. Write a loop that creates a column vector called Cprime which contains only the elements of C that are between -5 and 5 (inclusive).
    ## Create a vector called Cprime2 which does this calculation without a loop.
    Cprime = [] # GG: creating empty array
    for j=1:size(C,2)
        for i=1:size(C,1)
            if C[i,j]>=-5 && C[i,j]<=5 ## GG: declaring the IF condition here; note usage of && coz boolean here
                push!(Cprime,C[i,j]) ## GG: sounds like some sort of "copy"; "pushes" only element satisfying IF clause
            end
        end
    end
    Cprime2 = C[(C.>=-5) .& (C.<=5)]

    ## c. Using loops or comprehensions, create a 3-dimensional array called X that is of dimension N × K × T where
    ## N = 15,169, K = 6, and T = 5. For all t, the columns of X should be (in order):
    ## - an intercept (i.e. vector of ones)
    ## - a dummy variable that is 1 with probability .75∗(6−t)/5
    ## - a continuous variable distributed normal with mean 15+t−1 and standard deviation 5(t−1)
        ## GG: apparently (mean).+(sd).*randn(N,1) generates a normal rv with mean and sd
    ## - a continuous variable distributed normal with mean π(6−t)/3 and standard deviation 1/e
    ## - a discrete variable distributed “discrete normal” with mean 12 and standard deviation 2.19. (A discrete normal random variable is properly called a binomial random variable. The distribution described above can be implemented by choosing binomial parameters n and p where n = 20 and p = 0.6. Use the following code (after loading Julia’s Distributions package) to generate this vector of X: rand(Binomial(20,0.6),N), where N is the length of the vector
    ## - a discrete variable distributed binomial with n = 20 and p = 0.5
    N = 15169
    K = 6
    T = 5
    #=
    GG: cat is for CONCATENATION
    WOW this is basically generating a dataset in one line :O
    The dataset has N=15169 observations, K=6 variables, and T=5 time periods
    =#
    X = cat([cat([ones(N,1) rand(N,1).<=(0.75*(6-t)/5) (15+t-1).+(5*(t-1)).*randn(N,1) (π*(6-t)/3).+(1/exp(1)).*randn(N,1) rand(Binomial(20,0.6),N) rand(Binomial(20,0.5),N)];dims=3) for t=1:T]...;dims=3) # discrete_normal binomial

    ## d. Use comprehensions to create a matrix β which is K × T and whose elements evolve across time in the following fashion:
    ## - 1,1.25,1.5,...
    ## - ln(t)
    ## - −√t
    ## - exp{t}-exp{t+1}
    ## - t
    ## - t/3
    β = vcat([cat([1+0.25*(t-1) log(t) -sqrt(t) exp(t)-exp(t+1) t t/3];dims=1) for t=1:T]...)' ## GG: using vertical concatenation here; mind the transpose

    ## e. Use comprehensions to create a matrix Y which is N × T deﬁned by Y_t= X_t β_t + ε_t , where ε_t ∼ iid N (0,σ=.36)
    Y = hcat([cat(X[:,:,t]*β[:,t] + .36*randn(N,1);dims=2) for t=1:T]...) # GG: 3-dots at the end have to do with the layout given by hcat
    return nothing
end

#=
f. Wrap a function deﬁnition around all of the code for question 2. Call the function q2().
The function should have take as inputs the arrays A, B and C. It should return nothing.
At the very bottom of your script you should add the code q2(A,B,C). Make sure q2() gets called after q1()!
=#

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function q3()
    # a.
    nlsw88 = CSV.read("nlsw88.csv")
    save("nlsw88.jld","nlsw88",nlsw88)

    # b.
    mean(nlsw88.never_married)
    mean(nlsw88.collgrad)

    # c.
    freqtable(nlsw88, :race)

    # d.
    summarystats = describe(nlsw88)

    # e.
    freqtable(nlsw88, :industry, :occupation)

    # f.
    wageonly = nlsw88[:,[:industry,:occupation,:wage]]
    grouper = groupby(wageonly, [:industry,:occupation])
    combine(grouper, valuecols(grouper) .=> mean)
    return nothing
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 4
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function q4()
    # a.
    mats = load("firstmatrix.jld")
    A = mats["A"]
    B = mats["B"]
    C = mats["C"]
    D = mats["D"]

    # b.
    function matrixops(m1,m2)
        # c.
        # this function computes the following matrix formulas: i) element-wise multiplication; ii) transpose multiplication; iii) sum of the elementwise product
        # e.
        if size(m1)!=size(m2)
            error("inputs must have the same size.")
        end
        ret1 = m1.*m2
        ret2 = m1'*m2
        ret3 = sum(m1+m2)
        return ret1,ret2,ret3
    end

    # d.
    matrixops(A,B)

    # f.
    # matrixops(C,D)

    # g.
    mat1 = load("nlsw88.jld")
    nlsw88 = mat1["nlsw88"]
    matrixops(convert(Array,nlsw88.ttl_exp),convert(Array,nlsw88.wage))
    return nothing
end

# Call the functions defined above
A,B,C,D = q1()
q2(A,B,C)
q3()
q4()
