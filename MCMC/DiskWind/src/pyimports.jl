#!/usr/bin/env julia
module pyimports
    using PyCall
    G1D = Ref{PyObject}() #can't just declare empty variable with type pyObject, so instead declare empty reference
    pickle = Ref{PyObject}() #then access the reference with [] to get the actual PyObject
    numpy = Ref{PyObject}()
    function __init__() #need to only have one instance of these to prevent segfaults
        G1D[] = pyimport("scipy.ndimage").gaussian_filter1d
        pickle[] = pyimport("pickle")
        numpy[] = pyimport("numpy")
    end
    function readPickle(file)
        data = nothing
        @pywith pybuiltin("open")(file,"rb") as f begin
            data = pickle[].load(f,encoding="latin1")
        end
        return data
    end
    export G1D, pickle, readPickle
end


####TRY PUTTING THESE IN THEIR OWN MODULE TO STOP SEGFAULTS ######
#example 
# module Foo
#     using PyCall
#     const G = Ref{Float64}()
#     function __init__()
#        py"""
#        import astropy.constants as cons
#        """
#        G[] = py"float"(py"cons".G[])
#     end
# end
"""The __init__() function, if defined, is executed when the module is loaded.  When you have the @py stuff NOT inside __init__ it runs at precompilation time and the ephemeral values such as handles/pointers to the dynamically loaded python code are forever “baked in” to the module.  Unfortunately these values are invalid the next time you load the module so you end up with the segfault.
Putting the @py stuff inside __init__() ensures that the handles/pointers will be created for the current process at run time.  You’ll probably want to save the results of the @py stuff in variables that have wider scope."""
