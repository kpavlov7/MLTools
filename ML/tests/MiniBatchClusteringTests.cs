using ML.Clustering;
using System.Collections.Generic;
using Xunit;
using static Benchmark.BenchMetrics;

namespace ML.tests
{
    public class MiniBatchClusteringTests
    {
        private const double Epsilon = 1e-6;

        [Fact]
        public void dense_features_simple()
        {
            var instance1 = new[] { 20f, 30f, 0f, 0f };
            var instance2 = new[] { 20f, 30f, 0f, 0f };
            var instance3 = new[] { 0f, 0f, 10f, 20f };
            var instance4 = new[] { 0f, 0f, 10f, 20f };

            var inputFeaturesTypes = new InputFeatureTypes[] {
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Ordinal
            };

            var ordinalDenseSet = new InstanceRepresentation(inputFeaturesTypes, sparse : false);

            ordinalDenseSet.AddInstance(instance1);
            ordinalDenseSet.AddInstance(instance2);
            ordinalDenseSet.AddInstance(instance3);
            ordinalDenseSet.AddInstance(instance4);

            var clustering = new MiniBatchClustering(2, 3, 10);
            clustering.Train(ordinalDenseSet);
            var categories = clustering.Cluster(ordinalDenseSet);
        }

        [Fact]
        public void sparse_features_simple()
        {
            var instance1 = new[] { 20f, 30f, 0f, 0f };
            var instance2 = new[] { 20f, 30f, 0f, 0f };
            var instance3 = new[] { 0f, 0f, 10f, 20f };
            var instance4 = new[] { 0f, 0f, 10f, 20f };

            var inputFeaturesTypes = new InputFeatureTypes[] {
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Ordinal
            };

            var ordinalSparseSet = new InstanceRepresentation(
                inputFeaturesTypes,
                sparse: true);

            ordinalSparseSet.AddInstance(instance1);
            ordinalSparseSet.AddInstance(instance2);
            ordinalSparseSet.AddInstance(instance3);
            ordinalSparseSet.AddInstance(instance4);

            var clustering = new MiniBatchClustering(2, 3, 10);
            clustering.Train(ordinalSparseSet);
            var categories = clustering.Cluster(ordinalSparseSet);
        }

        [Fact]
        public void sparse_features_medium()
        {
            // Cluster generator settings
            var maxRadius = 100;
            var minRadius = 10;
            var clusterCount = 10;

            // Data size
            var featureDim = 100;
            var obsCount = 2000;

            var inputFeaturesTypes = new InputFeatureTypes[featureDim];

            for (var i = 0; i < featureDim; i++)
            {
                inputFeaturesTypes[i] = InputFeatureTypes.Ordinal;
            }

            var sparseOrdinalSet = new InstanceRepresentation(
                inputFeaturesTypes,
                sparse: true);

            var clusterGenerator = new SyntheticDataGenerator(maxRadius, minRadius, obsCount, clusterCount, featureDim);
            var trueClusterLabels = new List<int>();

            using (var obsGetter = clusterGenerator.GenerateClusterObservations().GetEnumerator())
            {
                var isNextObservation = obsGetter.MoveNext();

                while (isNextObservation)
                {
                    var cluster = obsGetter.Current.Item1;
                    var obs = obsGetter.Current.Item2;
                    trueClusterLabels.Add(cluster);
                    sparseOrdinalSet.AddInstance(obs);

                    isNextObservation = obsGetter.MoveNext();
                }
            }

            var clustering = new MiniBatchClustering(clusterCount, 100, 2000);
            clustering.Train(sparseOrdinalSet);
            var categories = clustering.Cluster(sparseOrdinalSet);

            var metricsGenerator = new MetricsGenerator();
            metricsGenerator.Add(Metrics.Purity);
            for( var i = 0; i < categories.Length; i++)
            {
                metricsGenerator.AddResult(categories[i], trueClusterLabels[i]);
                metricsGenerator.UpdateMetrics();
            }

            var purity = metricsGenerator.GetMetric(Metrics.Purity);

            Assert.True(purity + Epsilon - 0.6 > 0);
        }

        [Fact]
        public void dense_features_medium()
        {
            // Cluster generator settings
            var maxRadius = 100;
            var minRadius = 10;
            var clusterCount = 10;

            // Data size
            var featureDim = 100;
            var obsCount = 2000;

            var inputFeaturesTypes = new InputFeatureTypes[featureDim];

            for (var i = 0; i < featureDim; i++)
            {
                inputFeaturesTypes[i] = InputFeatureTypes.Ordinal;
            }

            var denseOrdinalSet = new InstanceRepresentation(inputFeaturesTypes, sparse: false);

            var clusterGenerator = new SyntheticDataGenerator(maxRadius, minRadius, obsCount, clusterCount, featureDim);
            var trueClusterLabels = new List<int>();

            using (var obsGetter = clusterGenerator.GenerateClusterObservations().GetEnumerator())
            {
                var isNextObservation = obsGetter.MoveNext();

                while (isNextObservation)
                {
                    var cluster = obsGetter.Current.Item1;
                    var obs = obsGetter.Current.Item2;
                    trueClusterLabels.Add(cluster);
                    denseOrdinalSet.AddInstance(obs);

                    isNextObservation = obsGetter.MoveNext();
                }
            }

            var clustering = new MiniBatchClustering(clusterCount, 100, 2000);
            clustering.Train(denseOrdinalSet);
            var categories = clustering.Cluster(denseOrdinalSet);

            var metricsGenerator = new MetricsGenerator();
            metricsGenerator.Add(Metrics.Purity);
            for (var i = 0; i < categories.Length; i++)
            {
                metricsGenerator.AddResult(categories[i], trueClusterLabels[i]);
                metricsGenerator.UpdateMetrics();
            }

            var purity = metricsGenerator.GetMetric(Metrics.Purity);

            Assert.True(purity + Epsilon - 0.6 > 0);
        }
    }
}