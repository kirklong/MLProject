module DiskWind
    function __init__()
        include("/home/kirk/Documents/research/Dexter/STORM/DiskWind/src/pyimports.jl")
        include("/home/kirk/Documents/research/Dexter/STORM/DiskWind/src/functions.jl")
        include("/home/kirk/Documents/research/Dexter/STORM/DiskWind/src/HSTutil.jl")  
    end
    export getProfiles, readPickle, getLCData, getSpectra, getHSTDataWrap
end
