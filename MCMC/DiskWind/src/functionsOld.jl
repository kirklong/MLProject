#!/usr/bin/env julia
using PyCall, Interpolations, Test, Statistics, StatsBase, Plots, Printf, BinnedStatistics
pickle = pyimport("pickle")
G1D = pyimport("scipy.ndimage").gaussian_filter1d

function readPickle(file)
    data = nothing
    @pywith pybuiltin("open")(file,"rb") as f begin
        data = pickle.load(f,encoding="latin1")
    end
    return data
end
#new script for building up more general Julia code

function get_rMinMax(r̄::Float64,rFac::Float64,γ::Float64)
    rMin = r̄*(rFac^(γ-5/2)-1)/(rFac^(γ-3/2)-1)*(γ-3/2)/(γ-5/2)
    rMax = rMin*rFac
    return rMin,rMax
end

meshgrid(x,y) = (repeat(x,outer=length(y)), repeat(y,inner=length(x)))

function setup(i::Float64,n1::Int64,n2::Int64,r̄::Float64,rFac::Float64,γ::Float64,coordsType::Symbol=:polar,scale::Symbol=:log)
    rMin,rMax = get_rMinMax(r̄,rFac,γ)
    i = i/180*π; cosi = cos(i); sini = sin(i) #inclination angle in rad
    α = nothing; β = nothing; r = nothing; ν = nothing; ϕ = nothing; dA = nothing
    if coordsType == :cartesian
        nx = n1; ny = n2; rlim = rMax
        a = nothing; b = nothing

        if scale == :linear
            a = range(-rlim,stop=rlim,length=nx); b = range(-rlim,stop=rlim,length=ny)
        elseif scale == :log
            a = vcat(-reverse(exp.(range(log(rMin*cosi),stop=log(rMax),length=Int(nx/2)))),exp.(range(log(rMin*cosi),stop=log(rMax),length=Int(nx/2))))
            b = vcat(-reverse(exp.(range(log(rMin*cosi),stop=log(rMax),length=Int(ny/2)))),exp.(range(log(rMin*cosi),stop=log(rMax),length=Int(ny/2))))
        else
            println("invalid scale symbol -- should be :linear or :log")
            exit()
        end

        α,β = meshgrid(a,b)
        α = reshape(α,nx,ny); β = reshape(β,nx,ny)

        dA = zeros(size(α))
        for i=1:size(dA)[1]
            for j=1:size(dA)[2]
                Δα = i<size(dA)[1] ? abs(α[i+1,j]-α[i,j]) : abs(α[end,j]-α[end-1,j]) #kinda bs but want n things and linear spacing so fine
                Δβ = j<size(dA)[2] ? abs(β[i,j+1]-β[i,j]) : abs(β[i,end]-β[i,end-1])
                dA[i,j] = Δα*Δβ
            end
        end

        r = reshape(sqrt.(β.^2 ./cosi^2 .+ α.^2),nx,ny); ϕ = reshape(atan.(β,α.*cosi),nx,ny)
        ν = 1 .+ sqrt.(1 ./(2 .*r)).*sini.*cos.(ϕ)

    elseif coordsType == :polar
        nr = n1; nϕ = n2
####################################################
        #offset = rand()*0.01
        #offset = exp(0.01) #irrational number, consistent
        offset = 0.
        ϕ = range(0+offset,stop=2π+offset,length=nϕ+1)[1:end-1] #so exclusive -- THIS IS THE PROBLEM
####################################################
        r = nothing; rGhost = nothing; Δr = nothing; Δlogr = nothing

        if scale == :linear
            r = range(rMin*cosi,stop=rMax,length=nr)
            Δr = r[2]-r[1]
            rGhost = [rMin*cosi-Δr,rMax*cosi+Δr]
        elseif scale == :log
            logr = range(log(rMin*cosi),stop=log(rMax),length=nr)
            Δlogr = logr[2]-logr[1]
            rGhost = exp.([log(rMin*cosi)-Δlogr,log(rMax)+Δlogr])
            r = exp.(logr)
        else
            println("invalid scale symbol -- should be :linear or :log")
            exit()
        end

        rMesh, ϕMesh = meshgrid(r,ϕ)
        rMesh = reshape(rMesh,nr,nϕ); ϕMesh = reshape(ϕMesh,nr,nϕ)
        α,β = rMesh.*cos.(ϕMesh), rMesh.*sin.(ϕMesh)
        #α,β = rMesh, ϕMesh
        Δϕ = ϕ[2]-ϕ[1]
        dA = zeros(size(rMesh))
        for i=1:size(dA)[1]
            for j=1:size(dA)[2]
                if scale == :log
                    Δr = exp(Δlogr*(i-1))-exp(Δlogr*(i-2)) #assuming min r = 1. i.e. min logr = 0.
                    #jason says this should just be r*Δlogr -- calculus -- but doing it that way is not as good when I test error?
                    Δr = rMesh[i,j]*Δlogr
                end
                dA[i,j] = rMesh[i,j]*Δϕ*Δr
            end
        end

        r = reshape(sqrt.(β.^2/cosi^2 .+ α.^2),nr,nϕ); ϕ = reshape(atan.(β,α*cosi),nr,nϕ)
        ν = 1 .+ sqrt.(1 ./(2 .* r)).*sini.*cos.(ϕ)

    else
        println("invalid coords system -- should be :cartesian or :polar")
        exit()
    end
    return reshape(α,n1,n2),reshape(β,n1,n2),r,ν,ϕ,sini,cosi,dA,rMin,rMax
end

getA(A0::Float64,r::Array{Float64,2},γ::Float64) = A0.*r.^γ

function dvldl(r::Array{Float64,2},sini::Float64,cosi::Float64,ϕ::Array{Float64,2},windWeight::Float64=0.,f1::Float64=1.,f2::Float64=1.,f3::Float64=1.)
    windφ1 = 3 .*sqrt.(1 ./(2 .*r))./r .*sini^2 .* cos.(ϕ).*(√2 .*cos.(ϕ).+sin.(ϕ)./2)
    windφ2 = cosi^2 .* (1 ./ (r.^(3/2))) # should be divided by H/R, so * R/H and say H/R ~ 0.01? leaving out for now because Jason said to -> absorbed into big constant
    windφ3 = -3 .*sqrt.(1 ./(2 .*r))./r .* sini*cosi .* cos.(ϕ)
    diskφ = 3 .*sqrt.(1 ./(2 .*r))./r .*sini^2 .* (cos.(ϕ).*sin.(ϕ)./2) #disk only
    dvl = (1-windWeight).*diskφ .+ windWeight.*(f1.*windφ1 .+ f2.*windφ2 .+ f3.*windφ3) #new terms approach
    return dvl
end

function intensity(A::Array{Float64,2},r::Array{Float64,2},dvldl::Array{Float64,2},τ::Float64; rMin::Float64=3e3,rMax::Float64=5e3,test::Bool=false)
    I = zeros(size(A))
    for i=1:size(A)[1]
        for j=1:size(A)[2]
             if test == true
                 rMin = 3e3; rMax = 5e3
                 I[i,j] = (r[i,j] > rMin && r[i,j] < rMax) ? 1. : 0.
             else
                 I[i,j] = (r[i,j] > rMin && r[i,j] < rMax) ? A[i,j]/(4π*r[i,j]^2) * abs(dvldl[i,j]) * (1. - exp(-τ)) : 0.
             end
         end
     end
     return I
 end

function getIntensity(r::Array{Float64,2},ϕ::Array{Float64,2},windWeight::Float64,sini::Float64,cosi::Float64,rMin::Float64,rMax::Float64,γ::Float64,A0::Float64,τ::Float64;
                        f1::Float64=1.,f2::Float64=1.,f3::Float64=1.,test::Bool=false)

    ϕ′ = ϕ .+ π/2
    ∇v = dvldl(r,sini,cosi,ϕ′,windWeight,f1,f2,f3)
    A = getA(A0,r,γ)
    I = intensity(A,r,∇v,τ,test=test,rMin=rMin,rMax=rMax)
    return I,γ,A0,τ
end

function plotIntensity(α::Array{Float64,},β::Array{Float64,},I::Array{Float64,})
    p = scatter(α,β,markerz=I.^(1/4),label="",ms=1,markeredgestrokewidth=0,marker=:dot,aspect_ratio=:equal)
    return p
end

function histSum(x::Array{Float64,},y::Array{Float64,};bins::Int=200,νMin::Float64,νMax::Float64)
    return binnedStatistic(x,y,nbins=bins,binMin=νMin,binMax=νMax)
end

function phase(ν::Array{Float64,2},I::Array{Float64,2},dA::Array{Float64,2},x::Array{Float64,2},
    y::Array{Float64,2},r::Array{Float64,2},U::Float64,V::Float64,rot::Float64,νMin::Float64,νMax::Float64,bins::Int=200)

    rot = rot/180*π
    u′ = cos(rot)*U+sin(rot)*V; v′ = -sin(rot)*U+cos(rot)*V
    dϕMap = -2*π*(x.*u′.+y.*v′).*I.*180/π*1e6
    edges,centers,dϕ = histSum(ν,dϕMap.*dA,bins=bins,νMin=νMin,νMax=νMax)
    edges,centers,iSum = histSum(ν,I.*dA,bins=bins,νMin=νMin,νMax=νMax)
    iSum[iSum .== 0.] .= 1.
    return dϕ./iSum
end

function getProfiles(params::Array{Float64,},data::Array{Array{Float64,N} where N, 1};
    bins::Int=200,nr::Int=1024,nϕ::Int=2048,coordsType::Symbol=:polar,scale_type::Symbol=:log,
    νMin = 0.98, νMax = 1.02) #corresponds to +/- 6 km/s by default
    #this is ~3x as fast as python version!

    i,r̄,Mfac,rFac,f1,f2,f3,pa,scale,cenShift = params; windWeight = 1.; γ = 1.; A0 = 1.; τ = 10. #some parameters fixed for now
    λCen = 2.172 + cenShift #microns, to compare with data
    #ν = (data[1].-2.172)./λCen #code units, cenShift should be small this is just for calculating min and max
    BLRAng = Mfac*3e8*2e33*6.67e-8/9e20/548/3.09e24 #solar masses * G / c^2 / Mpc -> end units = rad
    α,β,r,ν,ϕ,sini,cosi,dA,rMin,rMax = setup(i,nr,nϕ,r̄,rFac,γ,coordsType,scale_type)
    I,γ,A0,τ = getIntensity(r,ϕ,windWeight,sini,cosi,rMin,rMax,γ,A0,τ,f1=f1,f2=f2,f3=f3)
    νEdges,νCenters,flux = histSum(ν,I.*dA,bins=bins,νMin=νMin,νMax=νMax)

    UData = data[2]; VData = data[3]; psf=4e-3/2.35
    X = α.*BLRAng; Y = β.*BLRAng
    dϕList = []
    for i=1:length(UData)
        for ii in [I]
            dϕAvgRaw = phase(ν,ii,dA,X,Y,r,UData[i],VData[i],pa,νMin,νMax,bins)
            dϕAvg = G1D(dϕAvgRaw,psf/3e5/(νCenters[2]-νCenters[1]))
            push!(dϕList,dϕAvg)
        end
    end

    fline = flux./maximum(flux)*maximum(data[4])*scale
    lineAvg = G1D(fline,psf/3e5/(νCenters[2]-νCenters[1]))

    indx=[0,1,2,6,7,8,12,13,14,18,19,20].+1; oindx=[3,4,5,9,10,11,15,16,17,21,22,23].+1
    interpPhaseList = []; λ = λCen ./ νCenters #ν is really ν/ν_c -> ν/ν_c = (c/λ)/(c/λ_c) = λ_c/λ -> λ = λ_c / (ν/ν_c) = λ / code ν
    x = reverse(λ) #need to go from low to high for interpolation
    λData = data[1]
    for i=1:length(dϕList)
        yP = dϕList[i].*reverse(lineAvg) #so it matches x
        interpPhase = LinearInterpolation(x,yP,extrapolation_bc=Line())
        push!(interpPhaseList,interpPhase.(λData))
    end

    interpLine = LinearInterpolation(x,reverse(lineAvg),extrapolation_bc=Line())
    return λData, interpLine.(λData), interpPhaseList
end
