using ML.MathHelpers;
using System;
using Xunit;
using static ML.Instances;

namespace ML.tests
{
    public class InstanceRepresentationTests
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

            var ordinalDenseSet = new InstanceRepresentation(inputFeaturesTypes, sparse: false);

            ordinalDenseSet.AddInstance(instance1);
            ordinalDenseSet.AddInstance(instance2);
            ordinalDenseSet.AddInstance(instance3);
            ordinalDenseSet.AddInstance(instance4);

            Assert.Equal(instance2, ordinalDenseSet.Instances[1].GetValues());
            Assert.Equal(4, ordinalDenseSet.FeauturesCount);
        }

        [Fact]
        public void standardize_sparse_instance()
        {
            var instance1 = new[] { -1f, 5f, 0f, 1f };
            var instance2 = new[] { 0f, 6f, 1f, 1f };
            var instance3 = new[] { 3f, 7f, 1f, 1f };
            var instance4 = new[] { 4f, 0f, 1f, 0f };

            var feature1Sigma = Math.Sqrt(new[] { -1f, 0f, 3f, 4f }.Variance());
            var feature2Sigma = Math.Sqrt(new[] { 5f, 6f, 7f, 0f }.Variance());

            var feature1Mean = new[] { -1f, 0f, 3f, 4f }.Mean();
            var feature2Mean = new[] { 5f, 6f, 7f, 0f }.Mean();

            var inputFeaturesTypes = new InputFeatureTypes[] {
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Flags,
                InputFeatureTypes.Flags
            };

            var mixedSparseSet = new InstanceRepresentation(
                inputFeaturesTypes,
                sparse: true);

            mixedSparseSet.AddInstance(instance1);
            mixedSparseSet.AddInstance(instance2);
            mixedSparseSet.AddInstance(instance3);
            mixedSparseSet.AddInstance(instance4);

            mixedSparseSet.Standardize();

            var value1 = mixedSparseSet.Instances[0].GetValues();

            Assert.True(Math.Abs((-1f - feature1Mean) / feature1Sigma - value1[0]) < Epsilon);
            Assert.True((Math.Abs((5f - feature2Mean) / feature2Sigma - value1[1]) < Epsilon));
            Assert.Equal(-1f, value1[2]);
            Assert.Equal(1f, value1[3]);
        }

        [Fact]
        public void standardize_dense_instance()
        {
            var instance1 = new[] { -1f, 5f, 0f, 1f };
            var instance2 = new[] { 0f, 6f, 1f, 1f };
            var instance3 = new[] { 3f, 7f, 1f, 1f };
            var instance4 = new[] { 4f, 0f, 1f, 0f };

            var feature1Sigma = Math.Sqrt(new[] { -1f, 0f, 3f, 4f }.Variance());
            var feature2Sigma = Math.Sqrt(new[] { 5f, 6f, 7f, 0f }.Variance());

            var feature1Mean = new[] { -1f, 0f, 3f, 4f }.Mean();
            var feature2Mean = new[] { 5f, 6f, 7f, 0f }.Mean();

            var inputFeaturesTypes = new InputFeatureTypes[] {
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Flags,
                InputFeatureTypes.Flags
            };

            var mixedDenseSet = new InstanceRepresentation(
                inputFeaturesTypes,
                sparse: false);

            mixedDenseSet.AddInstance(instance1);
            mixedDenseSet.AddInstance(instance2);
            mixedDenseSet.AddInstance(instance3);
            mixedDenseSet.AddInstance(instance4);

            mixedDenseSet.Standardize();

            var value1 = mixedDenseSet.Instances[0].GetValues();

            Assert.True(Math.Abs((-1f - feature1Mean) / feature1Sigma - value1[0]) < Epsilon);
            Assert.True((Math.Abs((5f - feature2Mean) / feature2Sigma - value1[1]) < Epsilon));
            Assert.Equal(-1f, value1[2]);
            Assert.Equal(1f, value1[3]);
        }

        [Fact]
        public void rescale_dense_instance()
        {
            var instance1 = new[] { 1f, 5f, 0f, 1f };
            var instance2 = new[] { 0f, 6f, 1f, 1f };
            var instance3 = new[] { 3f, 7f, 1f, 1f };
            var instance4 = new[] { 4f, 0f, 1f, 0f };

            var feature1Max = 4f;
            var feature2Max = 7f;

            var feature1Min = 0f;
            var feature2Min = 0f;

            var inputFeaturesTypes = new InputFeatureTypes[] {
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Flags,
                InputFeatureTypes.Flags
            };

            var mixedDenseSet = new InstanceRepresentation(
                inputFeaturesTypes,
                sparse: false);

            mixedDenseSet.AddInstance(instance1);
            mixedDenseSet.AddInstance(instance2);
            mixedDenseSet.AddInstance(instance3);
            mixedDenseSet.AddInstance(instance4);

            mixedDenseSet.Rescale();

            var value1 = mixedDenseSet.Instances[0].GetValues();

            Assert.True(Math.Abs((1f - feature1Min) / feature1Max - value1[0]) < Epsilon);
            Assert.True((Math.Abs((5f - feature2Min) / feature2Max - value1[1]) < Epsilon));
            Assert.Equal(0, value1[2]);
            Assert.Equal(1f, value1[3]);
        }

        [Fact]
        public void rescale_sparse_instance()
        {
            var instance1 = new[] { 1f, 5f, 0f, 1f };
            var instance2 = new[] { 0f, 6f, 1f, 1f };
            var instance3 = new[] { 3f, 7f, 1f, 1f };
            var instance4 = new[] { 4f, 0f, 1f, 0f };

            var feature1Max = 4f;
            var feature2Max = 7f;

            var feature1Min = 0f;
            var feature2Min = 0f;

            var inputFeaturesTypes = new InputFeatureTypes[] {
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Ordinal,
                InputFeatureTypes.Flags,
                InputFeatureTypes.Flags
            };

            var mixedSparseSet = new InstanceRepresentation(
                inputFeaturesTypes,
                sparse: true);

            mixedSparseSet.AddInstance(instance1);
            mixedSparseSet.AddInstance(instance2);
            mixedSparseSet.AddInstance(instance3);
            mixedSparseSet.AddInstance(instance4);

            mixedSparseSet.Rescale();

            var value1 = mixedSparseSet.Instances[0].GetValues();

            Assert.True(Math.Abs((1f - feature1Min) / feature1Max - value1[0]) < Epsilon);
            Assert.True((Math.Abs((5f - feature2Min) / feature2Max - value1[1]) < Epsilon));
            Assert.Equal(0, value1[2]);
            Assert.Equal(1f, value1[3]);
        }

        [Fact]
        public void dist_sparse_instance()
        {
            var ordinalValues1 = new[] { 1f, 0f, 3f, 4f };
            var values1 = new[] { 1f, 3f, 4f };
            var indices1 = new[] { 0, 2, 3 };

            var binaryValues1 = new[] {1f, 0f, 1f, 0f};
            var binaryIndices1 = new[] { 0, 2 };

            var ordinalValues2 = new[] { 5f, 6f, 0f, 8f };
            var values2 = new[] { 5f, 6f, 8f };
            var indices2 = new[] { 0, 1, 3 };

            var binaryValues2 = new[] { 0f, 1f, 1f, 0f };
            var binaryIndices2 = new[] { 1, 2 };

            var t = new[,]
            {
                { 1f, 0f, 3f, 4f, 1f, 0f, 1f, 0f },
                { 5f, 6f, 0f, 8f, 0f, 1f, 1f, 0f }
            };

            var instance1 = new SparseInstance(
                    values1,
                    indices1,
                    binaryIndices1,
                    ordinalValues1.Length,
                    binaryValues1.Length);

            var instance2 = new SparseInstance(
                values2,
                indices2,
                binaryIndices2,
                ordinalValues2.Length,
                binaryValues2.Length);

            var distInstance1 = instance1.L2Dist(instance2);
            var distInstance2 = instance1.L2Dist(t, 1, 8);
            var distArray = t.L2Dist(0, 1, 8);

            Assert.True(Math.Abs(distInstance1 - distArray) < Epsilon);
            Assert.True(Math.Abs(distInstance2 - distArray) < Epsilon);
        }

        [Fact]
        public void dist_dense_instance()
        {
            var values1 = new[] { 1f, 0f, 3f, 4f, 1f, 0f, 1f, 0f };
            // index offset which indicates the start of the binary features
            var binaryOffset1 = 5;

            var values2 = new[] { 5f, 6f, 0f, 8f, 0f, 1f, 1f, 0f };
            // index offset which indicates the start of the binary features
            var binaryOffset2 = 5;

            var t = new[,]
            {
                { 1f, 0f, 3f, 4f, 1f, 0f, 1f, 0f },
                { 5f, 6f, 0f, 8f, 0f, 1f, 1f, 0f }
            };

            var instance1 = new DenseInstance(
                values1,
                binaryOffset1);

            var instance2 = new DenseInstance(
                values2,
                binaryOffset2);

            var distInstance1 = instance1.L2Dist(instance2);
            var distInstance2 = instance1.L2Dist(t, 1, 8);
            var distArray = t.L2Dist(0, 1, 8);

            Assert.True(Math.Abs(distInstance1 - distArray) < Epsilon);
            Assert.True(Math.Abs(distInstance2 - distArray) < Epsilon);
        }

        [Fact]
        public void get_value_instance()
        {
            var dValues = new[] { 1f, 0f, 3f, 4f, 1f, 0f, 1f, 0f };

            var sOrdinalValues = new[] { 1f, 0f, 3f, 4f };
            var dBinaryOffset = sOrdinalValues.Length;

            var sValues = new[] { 1f, 3f, 4f };
            var sIndices = new[] { 0, 2, 3 };

            var sBinaryValues = new[] { 1f, 0f, 1f, 0f };
            var sBinaryIndices = new[] { 0, 2 };

            var dInstance = new DenseInstance(
                dValues,
                dBinaryOffset);

            var sInstance = new SparseInstance(
                    sValues,
                    sIndices,
                    sBinaryIndices,
                    sOrdinalValues.Length,
                    sBinaryValues.Length);

            var sOutputValues = sInstance.GetValues();
            var dOutputValues = dInstance.GetValues();

            Assert.Equal(3f, sInstance.GetValue(2));
            Assert.Equal(3f, dInstance.GetValue(2));

            Assert.Equal(0f, sInstance.GetValue(7));
            Assert.Equal(0f, dInstance.GetValue(7));

            Assert.True(Math.Abs(dOutputValues.L2Norm() - sOutputValues.L2Norm()) < Epsilon);
        }
    }
}
