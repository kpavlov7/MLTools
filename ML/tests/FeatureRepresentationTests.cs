using System.Collections.Generic;
using System.Linq;
using Xunit;
using FT = ML.InputFeatureTypes;

namespace ML.tests
{
    public class FeatureRepresentationTests
    {
        private const double Epsilon = 1e-6;

        [Fact]
        public void not_sorted_features()
        {
            var feature1 = new[] { 20f, 30f,  0f, 0f }; // ordinal sparse
            var feature2 = new[] { 20f, 30f,  0f, 0f }; // ordinal dense
            var feature3 = new[] {  0f,  0f,  1f, 1f }; // flags
            var feature4 = new[] {  1f,  1f,  0f, 0f }; // flags
            var feature5 = new[] {  1f,  1f,  1f, 1f }; // flags
            var labels   = new[] {  0f,  1f, 12f, 1f };

            var notSortedFeatureRepresentation = new FeatureRepresentation();

            notSortedFeatureRepresentation.AddFeature(feature1, FT.Ordinal, isSparse: true);
            notSortedFeatureRepresentation.AddFeature(feature2, FT.Ordinal, isSparse: false);
            notSortedFeatureRepresentation.AddFeature(feature3, FT.Flags);
            notSortedFeatureRepresentation.AddFeature(feature4, FT.Flags);
            notSortedFeatureRepresentation.AddFeature(feature5, FT.Flags);

            notSortedFeatureRepresentation.AddLabels(labels);

            var features = notSortedFeatureRepresentation.Features;

            // We test the general paramters of the feature representation:
            Assert.Equal(
                new[] { FT.Ordinal, FT.Ordinal, FT.Flags, FT.Flags, FT.Flags },
                notSortedFeatureRepresentation.FeatureTypes);
            Assert.Equal(5, notSortedFeatureRepresentation.Features.Count);
            Assert.Equal(4, notSortedFeatureRepresentation.InstancesCount);
            Assert.Equal(
                new[] { 0, 1, 2, 1 },
                notSortedFeatureRepresentation.CategoricalLabels);
            Assert.Equal(
                new Dictionary<int,float>() { { 0, 0f }, { 1, 1f }, { 2, 12f } },
                notSortedFeatureRepresentation.TrueLabelsMap);

            // We test whether the feature representation preserves the values for
            // both sparse and dense:
            var representedFeature1 = features[0].GetValues();
            var representedFeature2 = features[1].GetValues();

            Assert.Equal(feature1, representedFeature1);
            Assert.Equal(representedFeature1, representedFeature2);

            // We test whether we preserve the binary features:
            Assert.Equal(feature3, features[2].GetValues());
            Assert.Equal(feature4, features[3].GetValues());
            Assert.Equal(feature5, features[4].GetValues());
        }

        [Fact]
        public void sorted_features()
        {
            var feature1 = new[] { 20f, 30f, 0f, 0f }; // ordinal sparse
            var feature2 = new[] { 20f, 30f, 0f, 0f }; // ordinal dense

            var sortedFeatureRepresentation = new FeatureRepresentation(isSortedDataset: true);

            sortedFeatureRepresentation.AddFeature(feature1, FT.Ordinal, isSparse: true);
            sortedFeatureRepresentation.AddFeature(feature2, FT.Ordinal, isSparse: false);

            var features = sortedFeatureRepresentation.Features;

            Assert.Equal(
                new[] { FT.Ordinal, FT.Ordinal},
                sortedFeatureRepresentation.FeatureTypes);
            Assert.Equal(2, features.Count);
            Assert.Equal(4, sortedFeatureRepresentation.InstancesCount);
            
            // We test whether the feature representation preserves the values for
            // both sparse and dense:
            var representedFeature1 = features[0].GetValues();
            var representedFeature2 = features[1].GetValues();

            Assert.Equal(representedFeature1, feature1);
            Assert.Equal(representedFeature1, representedFeature2);
        }
    }
}