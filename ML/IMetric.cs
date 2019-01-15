using static Benchmark.Metrics;

namespace Benchmark
{
    /// <summary>
    /// The Metric is used for assessing the performance of
    /// the Machine Learning models.
    /// </summary>
    public interface IMetric
    {
        double Get();
        void Add(MetricsInput metricInput);
    }
}
