# distutils: language = c++
# distutils: sources = densecrf/src/densecrf.cpp densecrf/src/unary.cpp
# densecrf/src/pairwise.cpp densecrf/src/permutohedral.cpp
# densecrf/src/optimization.cpp densecrf/src/objective.cpp
# densecrf/src/labelcompatibility.cpp densecrf/src/util.cpp
# densecrf/external/liblbfgs/lib/lbfgs.c
# distutils: include_dirs = densecrf/include densecrf/external/liblbfgs/include

from numbers import Number

import eigen
cimport eigen


cdef LabelCompatibility* _labelcomp(compat):
    if isinstance(compat, Number):
        return new PottsCompatibility(compat)
    elif memoryview(compat).ndim == 1:
        return new DiagonalCompatibility(eigen.c_vectorXf(compat))
    elif memoryview(compat).ndim == 2:
        return new MatrixCompatibility(eigen.c_matrixXf(compat))
    else:
        raise ValueError("LabelCompatibility of dimension >2 not meaningful.")


cdef class Unary:
    # Because all of the APIs that take an object of this type will
    # take ownership. Thus, we need to make sure not to delete this
    # upon destruction.
    cdef UnaryEnergy* move(self):
        ptr = self.thisptr
        self.thisptr = NULL
        return ptr

    # It might already be deleted by the library, actually.
    # Yeah, pretty sure it is.
    def __dealloc__(self):
        del self.thisptr


cdef class ConstUnary(Unary):
    def __cinit__(self, float[:,::1] u not None):
        self.thisptr = new ConstUnaryEnergy(eigen.c_matrixXf(u))


cdef class LogisticUnary(Unary):
    def __cinit__(self, float[:,::1] L not None, float[:,::1] f not None):
        self.thisptr = new LogisticUnaryEnergy(eigen.c_matrixXf(L),
                                               eigen.c_matrixXf(f))


cdef class LogLikelihoodObjective:
    def __cinit__(self, int[::1] gt not None, float robust=0):
        self.thisptr = new LogLikelihood(eigen.c_vectorXs(gt), robust)

    def __dealloc__(self):
        del self.thisptr


cdef class HammingObjective:
    def __cinit__(self, int[::1] gt not None, float class_weight_pow=0):
        self.thisptr = new Hamming(eigen.c_vectorXs(gt), class_weight_pow)

    # def __cinit__(self, int[::1] gt not None, float[::1] class_weight_pow not None):
    #     self.thisptr = new Hamming(eigen.c_vectorXs(gt),
    #                                eigen.c_vectorXf(class_weight_pow))

    def __dealloc__(self):
        del self.thisptr


# cdef class IntersectionOverUnionObjective:
#     def __cinit__(self, int[::1] gt not None):
#         self.thisptr = new IntersectionOverUnion(eigen.c_vectorXs(gt))
#
#     def __dealloc__(self):
#         del self.thisptr


cdef class CRFEnergy:
    def __cinit__(self, DenseCRF dc, str objective_name, int niter,
                  int[::1] labels, bint pairwise=True, bint kernel=True,
                  class_weight=0, float robust=0):
        if objective_name == 'Likelihood':
            objective = LogLikelihoodObjective(labels, robust)
        elif objective_name == 'Hamming':
            assert isinstance(class_weight, (float[::1], float)), \
                'class_weight has to be either numpy array or float'
            # if isinstance(class_weight, float[::1]):
            #     objective = HammingObjective(labels, class_weight)
            # elif isinstance(class_weight, float):
            #     objective = HammingObjective(eigen.c_vectorXs(labels), class_weight)
            # objective = HammingObjective(labels, class_weight)
        # elif objective_name == 'IoU':
        #     objective = IntersectionOverUnionObjective(labels)
        else:
            raise ValueError, 'Unknown objective function.'

        print type(objective)
        if type(self) is CRFEnergy:
            self.thisptr = new c_CRFEnergy(dc._this[0], objective.thisptr[0],
                                           niter, 0, int(pairwise), int(kernel))
        else:
            self.thisptr = NULL

        
    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            
    def setL2Norm(self, float norm):
        self.thisptr.setL2Norm(norm)
        
    # def gradient(self, float[::1] x, VectorXf dx):
    # def gradient(self, VectorXf x, VectorXf dx):
    #     self.thisptr.gradient(x.v, dx.v)
    def learn_parameters(self):
        p = minimizeLBFGS( self.thisptr[0], 2, 0 )
        return eigen.VectorXf().wrap(p)


cdef class DenseCRF:
    def __cinit__(self, int nvar, int nlabels, *_, **__):
        # We need to swallow extra-arguments because superclass cinit function
        # will always be called with the same params as the subclass, automatically.

        # We also only want to avoid creating an object if we're just being called
        # from a subclass as part of the hierarchy.
        if type(self) is DenseCRF:
            self._this = new c_DenseCRF(nvar, nlabels)
        else:
            self._this = NULL

    def __dealloc__(self):
        # Because destructors are virtual, this is enough to delete any object
        # of child classes too.
        if self._this:
            del self._this

    def addPairwiseEnergy(self, float[:,::1] features not None, compat,
                          KernelType kernel=DIAG_KERNEL,
                          NormalizationType normalization=NORMALIZE_SYMMETRIC):
        self._this.addPairwiseEnergy(eigen.c_matrixXf(features), _labelcomp(compat),
                                     kernel, normalization)

    def setUnary(self, Unary u):
        self._this.setUnaryEnergy(u.move())

    def setUnaryEnergy(self, float[:,::1] u not None, float[:,::1] f = None):
        if f is None:
            self._this.setUnaryEnergy(eigen.c_matrixXf(u))
        else:
            self._this.setUnaryEnergy(eigen.c_matrixXf(u), eigen.c_matrixXf(f))

    def inference(self, int niter):
        return eigen.MatrixXf().wrap(self._this.inference(niter))

    def startInference(self):
        return eigen.MatrixXf().wrap(self._this.startInference()), \
               eigen.MatrixXf(), eigen.MatrixXf()

    def stepInference(self, MatrixXf Q, MatrixXf tmp1, MatrixXf tmp2):
        self._this.stepInference(Q.m, tmp1.m, tmp2.m)

    def klDivergence(self, MatrixXf Q):
        return self._this.klDivergence(Q.m)

    def map(self, int niter):
        return eigen.VectorXs().wrap(self._this.map(niter))

    # def compute_gradient(self, int niter, int[::1] labels, float[::1] x,
    #                      float[::1] dx, str objective_name='Likelihood',
    #                      class_weight=0, float robust=0):
    #     if objective_name == 'Likelihood':
    #         objective = LogLikelihoodObjective(labels, robust)
    #     elif objective_name == 'Hamming':
    #         assert isinstance(class_weight, (float[::1], float)), \
    #             'class_weight has to be either numpy array or float'
    #         # if isinstance(class_weight, float[::1]):
    #         #     objective = Hamming(labels, class_weight)
    #         # elif isinstance(class_weight, float):
    #         #     objective = Hamming(eigen.c_vectorXs(labels), class_weight)
    #     #     objective = HammingObjective(labels, class_weight)
    #     elif objective_name == 'IoU':
    #         objective = IntersectionOverUnionObjective(labels)
    #     else:
    #         raise ValueError, 'Unknown objective function.'
    #
    #     # initial_u_param_ = self.unaryParameters * 0
    #     # initial_lbl_param_ = self.labelCompatibilityParameters # * 0
    #     # initial_knl_param_ = self.kernelParameters # * 0
    #     #
    #     # # du = eigen.c_vectorXf(initial_u_param_)
    #     # dl = eigen.c_vectorXf(initial_lbl_param_)
    #     # dk = eigen.c_vectorXf(initial_knl_param_)
    #
    #     dl = self._this.labelCompatibilityParameters() # * 0
    #     dk = self._this.kernelParameters()
    #
    #     grad_val = self._this.gradient(niter, objective.thisptr[0], NULL, &dl,
    #                                 &dk)

    property unaryParameters:
        def __get__(self):
#             c_VectorXf unaryParameters() const;
            return eigen.VectorXf().wrap(self._this.unaryParameters())
    
        def __set__(self, float[::1] value not None):
#             void setUnaryParameters( const c_VectorXf & v )
            self._this.setUnaryParameters(eigen.c_vectorXf(value))

    property labelCompatibilityParameters:
        def __get__(self):
#             c_VectorXf labelCompatibilityParameters() const
            return eigen.VectorXf().wrap(self._this.labelCompatibilityParameters())
    
        def __set__(self, float[::1] value not None):
#             void setLabelCompatibilityParameters( const c_VectorXf & v )
            self._this.setLabelCompatibilityParameters(eigen.c_vectorXf(value))
    
    property kernelParameters:
        def __get__(self):
#             c_VectorXf kernelParameters() const
            return eigen.VectorXf().wrap(self._this.kernelParameters())
    
        def __set__(self, float[::1] value not None):
#             void setKernelParameters( const c_VectorXf & v )
            self._this.setKernelParameters(eigen.c_vectorXf(value))

cdef class DenseCRF2D(DenseCRF):

    # The same comments as in the superclass' `__cinit__` apply here.
    def __cinit__(self, int w, int h, int nlabels, *_, **__):
        if type(self) is DenseCRF2D:
            self._this = self._this2d = new c_DenseCRF2D(w, h, nlabels)

    def addPairwiseGaussian(self, sxy, compat, KernelType kernel=DIAG_KERNEL,
                            NormalizationType normalization=NORMALIZE_SYMMETRIC):
        if isinstance(sxy, Number):
            sxy = (sxy, sxy)

        self._this2d.addPairwiseGaussian(sxy[0], sxy[1], _labelcomp(compat),
                                         kernel, normalization)

    def addPairwiseBilateral(self, sxy, srgb, unsigned char[:,:,::1] rgbim not None,
                             compat, KernelType kernel=DIAG_KERNEL,
                             NormalizationType normalization=NORMALIZE_SYMMETRIC):
        if isinstance(sxy, Number):
            sxy = (sxy, sxy)

        if isinstance(srgb, Number):
            srgb = (srgb, srgb, srgb)

        self._this2d.addPairwiseBilateral(sxy[0], sxy[1], srgb[0], srgb[1],
                                          srgb[2], &rgbim[0,0,0],
                                          _labelcomp(compat), kernel,
                                          normalization)
