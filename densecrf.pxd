from eigen cimport *


cdef extern from "densecrf/include/labelcompatibility.h":
    cdef cppclass LabelCompatibility:
        pass

    cdef cppclass PottsCompatibility(LabelCompatibility):
        PottsCompatibility(float) except +

    cdef cppclass DiagonalCompatibility(LabelCompatibility):
        DiagonalCompatibility(const c_VectorXf&) except +

    cdef cppclass MatrixCompatibility(LabelCompatibility):
        MatrixCompatibility(const c_MatrixXf&) except +


cdef extern from "densecrf/include/unary.h":
    cdef cppclass UnaryEnergy:
        pass

    cdef cppclass ConstUnaryEnergy(UnaryEnergy):
        ConstUnaryEnergy(const c_MatrixXf& unary) except +

    cdef cppclass LogisticUnaryEnergy(UnaryEnergy):
        LogisticUnaryEnergy(const c_MatrixXf& L, const c_MatrixXf& feature) except +


cdef class Unary:
    cdef UnaryEnergy *thisptr
    cdef UnaryEnergy* move(self)


cdef class ConstUnary(Unary):
    pass


cdef class LogisticUnary(Unary):
    pass


# I need to specify all that crap because of the abstract class and its
# virtual methods
cdef extern from "densecrf/include/pairwise.h":
    cpdef enum NormalizationType: NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC
    cpdef enum KernelType: CONST_KERNEL, DIAG_KERNEL, FULL_KERNEL


cdef extern from "densecrf/include/objective.h":
    cdef cppclass ObjectiveFunction:
        pass

    cdef cppclass LogLikelihood(ObjectiveFunction):
        LogLikelihood(const c_VectorXs & gt, float robust ) except +

    # cdef cppclass Hamming(ObjectiveFunction):
    #     Hamming(const c_VectorXs & gt, float class_weight_pow ) except +
    #     # Hamming(const c_VectorXs & gt, const c_VectorXf & class_weight ) \
    #     #         except +
    #
    # cdef cppclass IntersectionOverUnion(ObjectiveFunction):
    #     IntersectionOverUnion(const c_VectorXs & gt ) except +


cdef class LogLikelihoodObjective:
    cdef LogLikelihood *thisptr
#
#
# cdef class HammingObjective:
#     cdef Hamming *thisptr
#
#
# cdef class IntersectionOverUnionObjective:
#     cdef IntersectionOverUnion *thisptr


cdef extern from "densecrf/include/optimization.h":
    cdef cppclass c_EnergyFunction "EnergyFunction":
        pass
    cdef cppclass c_CRFEnergy "CRFEnergy" (c_EnergyFunction):
        c_CRFEnergy( c_DenseCRF & crf, const ObjectiveFunction & objective,
                     int NIT, bint unary, bint pairwise, bint kernel) except +
        void setL2Norm( float norm )
        double gradient( const c_VectorXf &x, c_VectorXf &dx )


cdef class CRFEnergy:
    cdef c_CRFEnergy *thisptr


cdef extern from "densecrf/include/optimization.h":
    c_VectorXf minimizeLBFGS( c_EnergyFunction & efun, int restart, bint verbose)


cdef extern from "densecrf/include/densecrf.h":
    cdef cppclass c_DenseCRF "DenseCRF":
        c_DenseCRF(int N, int M) except +

        # Setup methods.
        # TODO
        #void addPairwiseEnergy(PairwisePotential *potential)
        void addPairwiseEnergy(const c_MatrixXf &features, LabelCompatibility*,
                               KernelType, NormalizationType)
        void setUnaryEnergy(UnaryEnergy *unary)
        void setUnaryEnergy(const c_MatrixXf &unary)
        void setUnaryEnergy(const c_MatrixXf &L, const c_MatrixXf &feature)

        # Inference methods.
        c_MatrixXf inference(int n_iterations)
        # TODO: Not enabled because it would require wrapping VectorXs (note the `s`)
        c_VectorXs map(int n_iterations)

        # Step-by-step inference methods.
        c_MatrixXf startInference() const
        void stepInference(c_MatrixXf &Q, c_MatrixXf &tmp1, c_MatrixXf &tmp2) const
        double gradient( int n_iterations, const ObjectiveFunction & objective,
                         c_VectorXf * unary_grad,
                         c_VectorXf * lbl_cmp_grad,
                         c_VectorXf * kernel_grad ) const

        double klDivergence(const c_MatrixXf &Q) const

        c_VectorXf unaryParameters() const
        void setUnaryParameters( const c_VectorXf & v )
        c_VectorXf labelCompatibilityParameters() const
        void setLabelCompatibilityParameters( const c_VectorXf & v )
        c_VectorXf kernelParameters() const
        void setKernelParameters( const c_VectorXf & v )


cdef extern from "densecrf/include/densecrf.h":
    cdef cppclass c_DenseCRF2D "DenseCRF2D" (c_DenseCRF):
        c_DenseCRF2D(int W, int H, int M) except +

        void addPairwiseGaussian(float sx, float sy, LabelCompatibility*,
                                 KernelType, NormalizationType)
        void addPairwiseBilateral(float sx, float sy, float sr, float sg, float sb,
                                  const unsigned char *rgbim, LabelCompatibility*,
                                  KernelType, NormalizationType)


cdef class DenseCRF:
    cdef c_DenseCRF *_this


cdef class DenseCRF2D(DenseCRF):
    cdef c_DenseCRF2D *_this2d
