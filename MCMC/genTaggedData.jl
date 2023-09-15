#!/usr/bin/env julia
using Pkg
Pkg.activate("DiskWind")
using DiskWind, Distributions

function getModelProducts(θ,bins::Int=200,nr::Int=1024,nϕ::Int=2048)
    coordsType = :polar; scale_type = :log; νMin = 0.95; νMax = 1.05; τ = 10.
    i,r̄,Mfac,rFac,f1,f2,f3,f4,pa,Sα = θ
    s = f1 + f2 + f3 + f4
    f1,f2,f3,f4 = f1/s,f2/s,f3/s,f4/s

    α,β,r,ν,ϕ,sini,cosi,dA,rMin,rMax = DiskWind.setup(i,nr,nϕ,r̄,rFac,Sα,coordsType,scale_type)
    
    rs = 2*Mfac*3e8*2e30*6.67e-11/9e16
    days = 24*3600.
    t = r.*(rs/3e8/days) .* (1 .+ sin.(ϕ).*sini)
    ang = rs/(100e6*3.26*3e8*365*days)/4.848e-12 #μas size at fixed distance of 100 Mpc
    X = α.*ang; Y = β.*ang 

    I = DiskWind.getIntensity(r,ϕ,sini,cosi,rMin,rMax,Sα,τ,f1=f1,f2=f2,f3=f3,f4=f4)
    rBLR = sum(r.*I.*dA)/sum(I.*dA)*ang #rBLR in μas

    νEdges,νCenters,flux = DiskWind.histSum(ν,I.*dA,νMin=νMin,νMax=νMax,centered=true,bins=bins)
    νCenters = (νCenters.-1).*3e5 # convert to km/s
    LP = flux./maximum(flux) 

    t_char = sum(t.*I.*dA)/sum(I.*dA) # characteristic model size in days
    delays = zeros(length(νCenters))
    for i=1:length(νEdges)-1
        ν1,ν2 = νEdges[i],νEdges[i+1]
        mask = (ν.>ν1) .& (ν.<=ν2)
        delays[i] = sum(t[mask].*I[mask].*dA[mask])/sum(I[mask].*dA[mask])
    end

    Urange = range(-60,stop=0,length=10); Vrange = range(-60,stop=0,length=10) #Mλ, sparse coverage, only do "half" the box because symmetry about axis
    #initialize empty matrix where each entry is an array with length that matches νCenters
    phaseList = zeros(length(Urange),length(Vrange),length(νCenters))
    for i=1:length(Urange)
        for j=1:length(Vrange)
            dϕAvg = DiskWind.phase(ν,I,dA,α,β,r,Urange[i],Vrange[j],pa,νMin,νMax,bins)
            phaseList[i,j,:] .= dϕAvg .* LP ./ (1 .+ LP)
        end
    end
    meanPhase = [sum(phaseList[:,:,i]) for i=1:length(νCenters)]
    
    return νCenters,LP,meanPhase,delays,t_char,rBLR
end

function tagModelProducts(θ,bins::Int=200,nr::Int=1024,nϕ::Int=2048)
    νCenters,LP,meanPhase,delays,t_char,rBLR = getModelProducts(θ,bins,nr,nϕ)
    
    #identify single peak in LP
    LPmaxInd = argmax(LP)
    buffer = Int(round(bins/100))
    singlePeak = true; doublePeak = false
    if findfirst(LP.>0) == nothing || findlast(LP.>0) == nothing
        singlePeak = false
        doublePeak = false
        #error tag for no flux in line profile
        return singlePeak,doublePeak,"none",0,t_char,rBLR,-1
    end
    nonZeroL = LPmaxInd-findfirst(LP.>0); nonZeroR = findlast(LP.>0)-LPmaxInd
    centerOffset = buffer; quit = false
    while (centerOffset<(nonZeroL-buffer)) && (centerOffset<(nonZeroR-buffer)) && (quit == false)
        Δl = LP[LPmaxInd-centerOffset]-LP[LPmaxInd-(centerOffset+1)]; Δr = LP[LPmaxInd+centerOffset]-LP[LPmaxInd+(centerOffset+1)]
        if Δl<0 || Δr<0 #as we move away from center, LP should always decrease for single peak
            singlePeak = false
            doublePeak = true
            quit = true
        end
        centerOffset += 1
    end

    #identify rotation in S-curve in phase profile, record amplitude
    lMask = νCenters.<0; rMask = νCenters.>0
    lPhase = sum(meanPhase[lMask])/sum(lMask); rPhase = sum(meanPhase[rMask])/sum(rMask)
    rotation = "cw"
    if lPhase>rPhase
        rotation = "ccw"
    end
    phaseAmplitude = maximum(meanPhase)-minimum(meanPhase)

    #get FWHM of line profile
    lHalf = findfirst(LP.>0.5); rHalf = findlast(LP.>0.5)
    FHWM = νCenters[rHalf]-νCenters[lHalf] #km/s

    return singlePeak,doublePeak,rotation,phaseAmplitude,t_char,rBLR,FHWM
end
    
function genTaggedData(nSamples::Int,fOut = "tagged_samples.csv"; gridSearch = false,bins::Int=100,nr::Int=256,nϕ::Int=512)
    #generate random samples from parameter space
    s = time()
    open(fOut,"w") do f
        write(f,"i,r̄,Mfac,rFac,f1,f2,f3,f4,pa,Sα,singlePeak,doublePeak,rotation,phaseAmplitude,t_char,rBLR,FHWM\n")
    end
    if gridSearch
        nPerParam = Int(floor(nSamples^(1/10))) #number of samples per parameter, 10D parameter space
        if nPerParam < 2
            print("nSamples too small for grid search (need at least 2^10) reverting to random sampling\n")
            gridSearch = false
        else
            i = range(0,stop=90,length=nPerParam); r̄ = range(500,stop=5e4,length=nPerParam); Mfac = range(0.05,stop=5,length=nPerParam); rFac = range(2,stop=100,length=nPerParam); f1 = range(0,stop=1,length=nPerParam); f2 = range(0,stop=1,length=nPerParam); f3 = range(0,stop=1,length=nPerParam); f4 = range(0,stop=1,length=nPerParam); pa = range(0,stop=360,length=nPerParam); Sα = range(-2,stop=2,length=nPerParam)
            nSamples = nPerParam^10    
        end
        counter = 0
        for ii in i
            for r̄i in r̄
                for Mfaci in Mfac
                    for rFaci in rFac
                        for f1i in f1
                            for f2i in f2
                                for f3i in f3
                                    for f4i in f4
                                        for pai in pa
                                            for Sαi in Sα
                                                θ = [ii,r̄i,Mfaci,rFaci,f1i,f2i,f3i,f4i,pai,Sαi]
                                                singlePeak,doublePeak,rotation,phaseAmplitude,t_char,rBLR,FHWM = tagModelProducts(θ,bins,nr,nϕ)
                                                open(fOut,"a") do f
                                                    write(f,"$ii,$r̄i,$Mfaci,$rFaci,$f1i,$f2i,$f3i,$f4i,$pai,$Sαi,$singlePeak,$doublePeak,$rotation,$phaseAmplitude,$t_char,$rBLR,$FHWM\n")
                                                end
                                                counter+=1
                                                print(" "^100*"\r")
                                                print("$(round(100*counter/nSamples,sigdigits=2)) % complete\r")
                                                GC.gc()
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    if !gridSearch
        for n=1:nSamples
            i = rand(Uniform(0,90)); r̄ = rand(Uniform(500,5e4)); Mfac = rand(Uniform(0.05,5)); rFac = rand(Uniform(2,100)); f1 = rand(Uniform(0,1)); f2 = rand(Uniform(0,1)); f3 = rand(Uniform(0,1)); f4 = rand(Uniform(0,1)); pa = rand(Uniform(0,360)); Sα = rand(Uniform(-2,2))
            θ = [i,r̄,Mfac,rFac,f1,f2,f3,f4,pa,Sα]
            singlePeak,doublePeak,rotation,phaseAmplitude,t_char,rBLR,FHWM = tagModelProducts(θ,bins,nr,nϕ)
            open(fOut,"a") do f
                write(f,"$i,$r̄,$Mfac,$rFac,$f1,$f2,$f3,$f4,$pa,$Sα,$singlePeak,$doublePeak,$rotation,$phaseAmplitude,$t_char,$rBLR,$FHWM\n")
            end
            print(" "^100*"\r")
            print("$(round(100*n/nSamples,sigdigits=2)) % complete\r")
            GC.gc()
        end
    end
    f = time()
    println("$(round(f-s,sigdigits=2)) seconds to generate $nSamples samples")
end









