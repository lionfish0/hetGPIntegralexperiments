# Written by Mike Smith michaeltsmith.org.uk
from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
import math
import numpy as np
import math
import random

#TODO: Is it ok for us to just fill the rest of X in with zeros?
# these won't have any points chosen in those 0-volume areas... but ideally we should do something else? Put NaNs in????

#round (by probability)
#https://stackoverflow.com/questions/19045971/random-rounding-to-integer-in-python
def prob_round(x):
    sign = np.sign(x)
    x = abs(x)
    is_up = random.random() < x-int(x)
    round_func = math.ceil if is_up else math.floor
    return int(sign * round_func(x))


class ShapeIntegral(Kern):
    """
    
    """

    def __init__(self, input_dim, input_space_dim=None, active_dims=None, kernel=None, name='shapeintegral',Nperunit=100, lengthscale=[1.0], variance=1.0):
        """
        NOTE: Added input_space_dim as the number of columns in X isn't the dimensionality of the space. I.e. for pentagons there
        will be 10 columns in X, while only 2 dimensions of input space.
        
        The lengthscale, variance, etc are ideally set by specifying the kernel we'll use
        
        input_dim = number of actual columns in data
        input_space_dim = number of dimensions in the domain
        active_dims = potential list of dimensions we'll use
        kernel = latent function kernel
        Nperunit = resolution of approximation
        
        The last column of X should specify if it's the latent function or the integral that the Y refers to.
        if it's the latent function then we just use the first d-columns, and the rest can be NaN, e.g.
        X                               Y
        0,0,1,0,0,1,0,1,1,0,1,1,0     2
        1,1,nananananananananan,1     3
        is a 1x1 square with an integral of 2, and a single point in the [1,1] corner of the square with a value of 3.
        
        """
        super(ShapeIntegral, self).__init__(input_dim, active_dims, name)
        
        assert ((kernel is not None) or (input_space_dim is not None)), "Need either the input space dimensionality defining or the latent kernel defining (to infer input space)"
        if kernel is None:
            kernel = RBF(input_space_dim)
        else:
            input_space_dim = kernel.input_dim
        assert kernel.input_dim == input_space_dim, "Latent kernel (dim=%d) should have same input dimensionality as specified in input_space_dim (dim=%d)" % (kernel.input_dim,input_space_dim)
        
        
        #assert len(kern.lengthscale)==input_space_dim, "Lengthscale of length %d, but input space has %d dimensions" % (len(lengthscale),input_space_dim)

        #self.lengthscale = Param('lengthscale', kernel.lengthscale, Logexp()) #Logexp - transforms to allow positive only values...
        #self.variance = Param('variance', kernel.variance, Logexp()) #and here.
        #self.link_parameters(self.variance, self.lengthscale) #this just takes a list of parameters we need to optimise.
        
        self.kernel = kernel
        self.Nperunit = Nperunit
        self.input_space_dim = input_space_dim
        self.cached_points = {} #this is important, not only is it a speed up - we also get the same points for each shape, which makes our covariances more stable
        
        self.lengthscale = Param('lengthscale', lengthscale, Logexp()) #Logexp - transforms to allow positive only values...
        self.variance = Param('variance', variance, Logexp()) #and here.
        self.link_parameters(self.variance, self.lengthscale) #this just takes a list of parameters we need to optimise.

    def simplexRandom(self,vectors):
        #vectors = np.array([[0,0],[0,2],[1,1]])
        """
        Compute random point in arbitrary simplex

        from Grimme, Christian. Picking a uniformly
        random point from an arbitrary simplex.
        Technical Report. University of M\:{u}nster, 2015.

        vectors are row-vectors describing the
        vertices of the simplex, e.g.
        [[0,0],[0,2],[1,1]] is a triangle
        """
        d = vectors.shape[1]
        n = vectors.shape[0]
        assert n == d+1, "Need exactly d+1 vertices to define a simplex (e.g. a 2d triangle needs 3 points, a 3d tetrahedron 4 points, etc). Currently have %d points and %d dimensions" % (n,d)

        zs = np.r_[1,np.random.rand(d),0]
        ls = zs**(1.0/np.arange(len(zs)-1,-1,-1))
        vs = np.cumprod(ls) #could skip last element for speed
        res = vectors.copy()
        res = np.zeros(d)
        for vect,l,v in zip(vectors.copy(),ls[1:],vs):
            res+=(1-l)*v*vect
        return res

    def simplexVolume(self, vectors):
        """Returns the volume of the simplex defined by the
        row vectors in vectors, e.g. passing [[0,0],[0,2],[2,0]]
        will return 2 (as this triangle has area of 2)"""
        assert vectors.shape[0]==self.input_space_dim+1, "For a %d dimensional space there should be %d+1 vectors describing the simplex" % (self.input_space_dim, self.input_space_dim)
        return np.abs(np.linalg.det(vectors[1:,:]-vectors[0,:]))/math.factorial(self.input_space_dim)
        
    def delete_cache(self):
        self.cached_points = {}
      
    def polytopeVolume(self, shape):
        Ncoords = self.input_space_dim*(self.input_space_dim+1)
        vol = 0
        for i in range(0,len(shape),Ncoords):
            vectors = np.array(shape[i:(i+Ncoords)].reshape(self.input_space_dim+1,self.input_space_dim))
            if np.isnan(vectors[0,0]):
                break
            vol += self.simplexVolume(vectors)
        return vol
            
            
            
    def placepoints(self,shape,Nperunit=100):
        """Places uniformly random points in shape, where shape
        is defined by an array of concatenated simplexes
        e.g. a 2x2 square (from [0,0] to [2,2]) could be built
        of two triangles:
        [0,0,0,2,2,0 ,2,2,0,2,2,0]"""
        
        #caching is vital to the algorithm's computed covariance matrix
        #we don't want to cause non-positive-definite matrices
        tuplehash = hash(shape.data.tobytes()) #TODO Add Nperunit to hash
        if tuplehash in self.cached_points:
            return self.cached_points[tuplehash]
        allps = []
        #each simplex in shape must have D*(D+1) coordinates, e.g. a triangle has 2*(2+1) = 6 coords (2 for each vertex)
        #e.g. a tetrahedron has 4 points, each with 3 coords = 12: 3*(3+1) = 12.
        
        Ncoords = self.input_space_dim*(self.input_space_dim+1)
        assert len(shape)%Ncoords == 0, "The number of coordinates (%d) describing the simplexes that build the shape must factorise into the number of coordinates in a single simplex in %d dimensional space (=%d)" % (len(shape), self.input_space_dim, Ncoords)
        
        totalvol = 0
        vols = []
        for i in range(0,len(shape),Ncoords):
            vectors = np.array(shape[i:(i+Ncoords)].reshape(self.input_space_dim+1,self.input_space_dim))
            if np.isnan(vectors[0,0]):
                break
            vol = self.simplexVolume(vectors)
            totalvol += vol
            vols.append(vol)

        totalexpectednpoints = 0
        totalnpoints = 0
        for i,vol in zip(range(0,len(shape),Ncoords),vols):
            vectors = shape[i:(i+Ncoords)].reshape(self.input_space_dim+1,self.input_space_dim)
            if np.isnan(vectors[0,0]):
                break
            npoints = prob_round(Nperunit*vol/totalvol)
            totalexpectednpoints += Nperunit*vol/totalvol
            totalnpoints += npoints
      #      if (totalnpoints<totalexpectednpoints-1) and (vol>0):
      #          npoints+=1
      #      if (totalnpoints>totalexpectednpoints+1) and (npoints>=1):
      #          npoints-=1
                
            points = np.array([self.simplexRandom(vectors) for i in range(npoints)])
            allps.extend(points)
            
        #hack to ensure this polygon has a point
        if len(allps)==0:
            allps.append(self.simplexRandom(vectors))
        assert len(allps)>0, "No points were placed for this shape"
            
        self.cached_points[tuplehash] = (np.array(allps),totalvol)
        return np.array(allps),totalvol
        
    def k_xx(self,X,X2=None):
        """Calculates K_xx without the variance term
        
        X is in the form of an array *for one shape*. it
        is defined by an array of concatenated simplexes
        e.g. a 2x2 square (from [0,0] to [2,2]) could be built
        of two triangles:
        [0,0,0,2,2,0 ,2,2,0,2,2,0]
        """
        if X2 is None:
            X2 = X
        ps, vp = self.placepoints(X,self.Nperunit)
        qs, vq = self.placepoints(X2,self.Nperunit)
        if len(ps)==0 or len(qs)==0: return 0
        cov = self.kernel.K(ps,qs)
        v = vp*vq*np.sum(cov)/np.prod(cov.shape) #(self.Nperunit*self.Nperunit)
        return v
    
    def dkd_xx(self,X,X2=None):
        """Calculates dK_xx/dl without the variance term
        
        X is in the form of an array *for one shape*. it
        is defined by an array of concatenated simplexes
        e.g. a 2x2 square (from [0,0] to [2,2]) could be built
        of two triangles:
        [0,0,0,2,2,0 ,2,2,0,2,2,0]
        """
        if X2 is None:
            X2 = X
        ps, vp = self.placepoints(X,self.Nperunit)
        qs, vq = self.placepoints(X2,self.Nperunit) 
        if len(ps)==0 or len(qs)==0: return 0,0
        self.kernel.update_gradients_full(np.ones([len(ps),len(qs)]),ps,qs)
        dl = vp*vq*self.kernel.lengthscale.gradient/(len(ps)*len(qs))
        dv = vp*vq*self.kernel.variance.gradient/(len(ps)*len(qs))
        
        return dl,dv
        
    def k_xf(self,X,X2):
        """Calculates K_xf without the variance term
        
        X2 is in the form of an single row, one shape. It
        is defined by an array of concatenated simplexes
        e.g. a 2x2 square (from [0,0] to [2,2]) could be built
        of two triangles:
        [0,0,0,2,2,0 ,2,2,0,2,2,0]
        
        X is a list of coordinates.
        """
        
        #assert X.shape[1]==self.kernel.input_dim, "First term (X) should be locations of points in latent space. Second term should be shapes of integrals."

        ps, vp = self.placepoints(X2,self.Nperunit)
        if len(ps)==0: return 0
        cov = self.kernel.K(ps,X[None,0:self.input_space_dim])
        v = vp*np.sum(cov)/np.prod(cov.shape) #(self.Nperunit*self.Nperunit)
        return v
    
    def dkd_xf(self,X,X2):
        """Calculates dK_xf/dl
        
        X2 is in the form of an single row, one shape. It
        is defined by an array of concatenated simplexes
        e.g. a 2x2 square (from [0,0] to [2,2]) could be built
        of two triangles:
        [0,0,0,2,2,0 ,2,2,0,2,2,0]
        
        X is a list of coordinates.
        """
        
        #assert X.shape[1]==self.kernel.input_dim, "First term (X) should be locations of points in latent space. Second term should be shapes of integrals."

        ps, vp = self.placepoints(X2,self.Nperunit)
        if len(ps)==0: return 0,0
       #print(np.ones([len(ps),1]),ps,X[None,0:self.input_space_dim])
        self.kernel.update_gradients_full(np.ones([len(ps),1]),ps,X[None,0:self.input_space_dim])
        dl = vp*self.kernel.lengthscale.gradient/(len(ps))
        dv = vp*self.kernel.variance.gradient/(len(ps))
        return dl,dv
    
    def k_ff(self,X,X2):
        return self.kernel.K(X[None,0:self.input_space_dim],X2[None,0:self.input_space_dim])
    
    def dkd_ff(self,X,X2):
        #print(np.ones([X.shape[0],X2.shape[0]]),X[None,0:self.input_space_dim],X2[None,0:self.input_space_dim])
        self.kernel.update_gradients_full(np.ones([1,1]),X[None,0:self.input_space_dim],X2[None,0:self.input_space_dim])
        #print("LENGTHSCALE GRADIENT")
        #print(self.kernel.lengthscale.gradient)
        return self.kernel.lengthscale.gradient,self.kernel.variance.gradient
       

    def k(self,x,x2):
        """Helper function to compute covariance in one dimension (idx) between a pair of points.
        The last element in x and x2 specify if these are integrals (0) or latent values (1).
        l = that dimension's lengthscale
        """
        
        if (x[-1]==0) and (x2[-1]==0):
            return self.k_xx(x[:-1],x2[:-1])
        if (x[-1] == 1) and (x2[-1] == 0):
            return self.k_xf(x[:-1],x2[:-1])
        if (x[-1] == 0) and (x2[-1] == 1):
            return self.k_xf(x2[:-1],x[:-1])
        if (x[-1]==1) and (x2[-1]==1):
            return self.k_ff(x[:-1],x2[:-1])
        assert False, "Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)"
        
    def dkd(self,x,x2):
        """Helper function to compute gradient of K wrt lengthscale"""
        if (x[-1]==0) and (x2[-1]==0):
            return self.dkd_xx(x[:-1],x2[:-1])
        if (x[-1] == 1) and (x2[-1] == 0):
            return self.dkd_xf(x[:-1],x2[:-1])
        if (x[-1] == 0) and (x2[-1] == 1):
            return self.dkd_xf(x2[:-1],x[:-1])
        if (x[-1]==1) and (x2[-1]==1):
            return self.dkd_ff(x[:-1],x2[:-1])
        assert False, "Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)"
    
    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        Not solved - this needs doing (pass the results from the latent kernel)
        """
        if X2 is None:
            X2 = X
        
        dK_dl = np.zeros([X.shape[0],X2.shape[0]]) #ones now as a product occurs over each dimension
        dK_dv = np.zeros([X.shape[0],X2.shape[0]]) #ones now as a product occurs over each dimension
        for i,x in enumerate(X):
            for j,x2 in enumerate(X2):
                dl,dv = self.dkd(x,x2)
                dK_dl[i,j] += dl
                dK_dv[i,j] += dv
        #TODO - multiple lengthscales
        self.lengthscale.gradient = np.sum(dL_dK * dK_dl) # * self.variances[0] #TODO - is this ok???
        self.variance.gradient = np.sum(dL_dK * dK_dv) # * self.variances[0] #TODO - is this ok???

    def update_gradients_diag(self, dL_dKdiag, X):
        """
        Not solved - this needs doing (pass the results from the latent kernel)
        """
        pass
            
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        K_xx = np.ones([X.shape[0],X2.shape[0]]) #ones now as a product occurs over each dimension
        for i,x in enumerate(X):
            for j,x2 in enumerate(X2):
                #for il,l in enumerate(self.kernel.lengthscale):
                #    idx = il*2 #each pair of input dimensions describe the limits on one actual dimension in the data
                K_xx[i,j] *= self.k(x,x2)
        return K_xx# * self.variances[0] #TODO - is this ok???
            

    def Kdiag(self, X):
        return np.diag(self.K(X))
    
    def parameters_changed(self):
        self.kernel.lengthscale = self.lengthscale
        self.kernel.variance = self.variance
        
    def gradients_X(self, dL_dK, X, X2=None):
        #     """
        #     .. math::
        #         \\frac{\partial L}{\partial X} = \\frac{\partial L}{\partial K}\\frac{\partial K}{\partial X}
        #     """
        pass #TODO!
        #self.kernel.gradients_X
