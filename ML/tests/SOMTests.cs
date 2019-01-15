using Xunit;
using System.Collections.Generic;
using static Benchmark.Metrics;

namespace ML.tests
{
    public class SOMTests
    {
        private const double Epsilon = 1e-6;

        [Fact]
        public void to_coordinates()
        {
            ushort x; ushort n = 4;
            ushort y; ushort m = 5;

            var index = 14;

            var k = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    if (k == index)
                    {
                        x = (ushort)i;
                        y = (ushort)j;
                    }
                    k++;
                }
            }

            var som = new SelfOrganizingMap(1, new ushort[] { n, m }, 1, 1);
            var coordinates = som.ToCoordinates(index);
        }
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

            var ordinalDenseSet = new InstanceRepresentation(inputFeaturesTypes, sparse: false);

            ordinalDenseSet.AddInstance(instance1);
            ordinalDenseSet.AddInstance(instance2);
            ordinalDenseSet.AddInstance(instance3);
            ordinalDenseSet.AddInstance(instance4);

            var som = new SelfOrganizingMap(10, new ushort[] { 20, 20 }, 100, 3);

            var categories = som.Train(ordinalDenseSet);
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

            var sparseOrdinalSet = new InstanceRepresentation(inputFeaturesTypes, sparse: true);

            var clusterGenerator = new SyntheticDataGenerator(maxRadius, minRadius, obsCount, clusterCount, featureDim);
            var trueClusterLabels = new List<int>();

            using (var obsGetter = clusterGenerator.GenerateClusterObservations().GetEnumerator())
            {
                var isNextObservation = obsGetter.MoveNext();

                while (isNextObservation)
                {
                    var obs = obsGetter.Current.Item2;
                    var cluster = obsGetter.Current.Item1;
                    trueClusterLabels.Add(cluster);
                    sparseOrdinalSet.AddInstance(obs);

                    isNextObservation = obsGetter.MoveNext();
                }
            }

            var som = new SelfOrganizingMap(10, new ushort[] { (ushort)(clusterCount + 5), (ushort)(clusterCount + 5) }, 1000, 3);
            var categories = som.Train(sparseOrdinalSet);

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
                    var obs = obsGetter.Current.Item2;
                    var cluster = obsGetter.Current.Item1;
                    trueClusterLabels.Add(cluster);
                    denseOrdinalSet.AddInstance(obs);

                    isNextObservation = obsGetter.MoveNext();
                }
            }

            var som = new SelfOrganizingMap(10, new ushort[] { (ushort)(clusterCount + 5), (ushort)(clusterCount + 5)}, 1000, 3);
            var categories = som.Train(denseOrdinalSet);

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
