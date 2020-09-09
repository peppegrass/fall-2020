# problem set 1 solutions

using DataFrames
using FreqTables
using CSV
using JLD
using Random
using LinearAlgebra
using Statistics

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# a. Set the seed
Random.seed!(1234)

function q1()
    # i.-iv. create matrices
    A = 15*rand(10,7).-5
    B = 15*randn(10,7).-2
    C = cat(A[1:5,1:5],B[1:5,end-1:end]; dims=2)
    D = A.*(A.<=0)

    # b. number of elements
    println(length(A))

    # c. number of unique elements
    println(length(unique(A)))

    # d. reshape (vec operator)
    # one way:
    E = reshape(B,length(B))

    # easy way:
    E = B[:]

    # e. 3-dim array
    F = cat(A,B; dims=3)

    # f. permute
    F = permutedims(F, [3 1 2])

    # g. Kronecker
    G = kron(B,C)
    #kron(C,F)

    # h. saving
    save("matrixpractice.jld","A",A,"B",B,"C",C,"D",D,"E",E,"F",F,"G",G)

    # i. saving
    save("firstmatrix.jld","A",A,"B",B,"C",C,"D",D)

    # j. export to CSV
    CSV.write("Cmatrix.csv",DataFrame(C))

    # k. export to DAT
    CSV.write("Dmatrix.dat",DataFrame(D); delim="\t")
    return A,B,C,D
end
#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#:::::::::::::::::::::::::::::::::::::::::::::::::::

function q2(A,B,C)
    # a.
    AB = [A[i,j]*B[i,j] for i=1:size(A,1),j=1:size(A,2)]
    AB2 = A.*B

    # b.
    Cprime = []
    for j=1:size(C,2)
        for i=1:size(C,1)
            if C[i,j]>=-5 && C[i,j]<=5
                push!(Cprime,C[i,j])
            end
        end
    end
    Cprime2 = C[(C.>=-5) .& (C.<=5)]

    # c.
    N = 15_169
    K = 6
    T = 5
    X = cat([cat([ones(N,1) rand(N,1).<=(0.75*(6-t)/5) (15+t-1).+(5*(t-1)).*randn(N,1) (π*(6-t)/3).+(1/exp(1)).*randn(N,1) rand(Binomial(20,0.6),N) rand(Binomial(20,0.5),N)];dims=3) for t=1:T]...;dims=3) # discrete_normal binomial

    # d.
    β = vcat([cat([1+0.25*(t-1) log(t) -sqrt(t) exp(t)-exp(t+1) t t/3];dims=1) for t=1:T]...)'

    # e.
    Y = hcat([cat(X[:,:,t]*β[:,t] + .36*randn(N,1);dims=2) for t=2:T]...)
    return nothing
end

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