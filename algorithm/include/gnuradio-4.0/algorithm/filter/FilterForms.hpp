#ifndef GNURADIO_ALGORITHM_FILTER_FORMS_HPP
#define GNURADIO_ALGORITHM_FILTER_FORMS_HPP

namespace gr::algorithm::filter {

/// which state a recursion keeps between samples, and where: direct form II folds the two histories into one,
/// the transposed forms carry the accumulators instead of the samples
enum class IIRForm { DF_I, DF_II, DF_I_TRANSPOSED, DF_II_TRANSPOSED };

/// where a convolution is evaluated
enum class ConvolutionDomain { Auto, Time, Frequency };

/// whether a design carries feedback; an FIR is the special case whose feedback is `{1}`
enum class FilterType { FIR, IIR };

} // namespace gr::algorithm::filter

#endif // GNURADIO_ALGORITHM_FILTER_FORMS_HPP
