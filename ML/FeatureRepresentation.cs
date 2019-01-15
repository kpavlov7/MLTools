using ML.MathHelpers;
using System;
using System.Collections.Generic;
using System.Linq;
using static ML.Features;

namespace ML
{
    /// <summary>
    /// The Feature Representation is intended as a dataset implementation in suitable
    /// format for Machine Learning.
    /// </summary>
    public class FeatureRepresentation
    {
        //TODO: remove InstancesCount since it can be derived from Features
        //TODO: or CategoricalLabels
        public int InstancesCount { get; set; }
        public List<IFeature> Features { get; }
        /// <summary>
        /// Labels expressing the category levels of the 
        /// the true label values and they are in the range
        /// {0, 1, 2, ...}, so that it can be used directly
        /// as indices. These labels are used for classification
        /// algorithms.
        /// </summary>
        public int[] CategoricalLabels { get; set; }
        public bool IsSortedDataset { get; }

        //TODO: FeatureTypes and _trueLabelsMap are algorithm specific info,
        //TODO: thus we need to keep that information on ML algorithm level.
        public List<InputFeatureTypes> FeatureTypes { get; }
        public Dictionary<int, float> TrueLabelsMap { get; set; }

        public FeatureRepresentation(bool isSortedDataset = false)
        {
            IsSortedDataset = isSortedDataset;
            InstancesCount = 0;
            FeatureTypes = new List<InputFeatureTypes>();
            Features = new List<IFeature>();
            var ordinalIndices = new List<int>();
            var flagsIndices = new List<int>();
        }

        public FeatureRepresentation(
            int instancesCount,
            List<IFeature> features,
            List<InputFeatureTypes> featureTypes,
            Dictionary<int, float> trueLabelsMap,
            int[] categoricalLabels,
            bool isSortedDataset)
        {
            InstancesCount = instancesCount;
            Features = features;
            FeatureTypes = featureTypes;
            TrueLabelsMap = trueLabelsMap;
            CategoricalLabels = categoricalLabels;
            IsSortedDataset = isSortedDataset;
        }

        // TODO: framework for checking if the feature values are mmeting requirements.
        /// <summary> Adds feature(column) to the feature representation. </summary>
        /// <param name="isSparse"> 
        /// It doesn't affect the flags, they are always sparse.
        /// </param>
        public void AddFeature(
            float[] values, InputFeatureTypes type,
            bool isSparse = false)
        {
            FeatureTypes.Add(type);
            var sIndices = new List<int>();

            if (InstancesCount != 0)
            {
                if (InstancesCount != values.Length)
                    throw new ArgumentException("The feature length is different than instances count.");
            }
            else
                InstancesCount = values.Length;

            switch (type)
            {
                case InputFeatureTypes.Ordinal:
                    if (IsSortedDataset)
                    {
                        if (isSparse)
                        {
                            SortSparseInput(
                                values,
                                out var nonZeroSortedValues,
                                out var nonZeroSortedIndices,
                                out var negCount);

                            Features.Add(new SparseFeature(
                                values: nonZeroSortedValues,
                                sortedIndices: nonZeroSortedIndices,
                                length: values.Length,
                                scopeOffset: 0,
                                scopeCount: nonZeroSortedIndices.Length,
                                negCount: negCount));
                            break;
                        }
                        else
                        {
                            var sortedIndices = new int[values.Length];
                            for (var i = 0; i < values.Length; i++)
                            {
                                sortedIndices[i] = i;
                            }
                            Array.Sort(values, sortedIndices);
                            Features.Add(new DenseFeature(values, sortedIndices, 0, values.Length));
                            break;
                        }
                    }
                    if (isSparse)
                    {
                        var sValues = new List<float>();
                        for(var i = 0; i < values.Length; i++)
                        {
                            if(values[i] > 0f)
                            {
                                sValues.Add(values[i]);
                                sIndices.Add(i);
                            }
                        }
                        Features.Add(new SparseFeature(
                            values: sValues.ToArray(),
                            sortedIndices: sIndices.ToArray(),// Not sorted case;
                            length: values.Length,
                            scopeOffset: 0,
                            scopeCount: sIndices.Count,
                            negCount: -1));// Not sorted case;
                        break;
                    }
                    var copiedValues = new float[values.Length];
                    Array.Copy(values, copiedValues, values.Length);

                    Features.Add(new DenseFeature(values));
                    break;

                case InputFeatureTypes.Flags:
                    for (var i = 0; i < values.Length; i++)
                    {
                        if (values[i] > 0f)
                        {
                            sIndices.Add(i);
                        }
                    }
                    Features.Add(new BinaryFeature(sIndices.ToArray(), values.Length, 0, sIndices.Count));
                    break;
            }
        }

        public void AddLabels(float[] labels)
        {
            TrueLabelsMap = new Dictionary<int, float>();
            CategoricalLabels = new int[labels.Length];
            var categoriesMap = new Dictionary<float, int>();

            if (InstancesCount != 0)
            {
                if (InstancesCount != labels.Length)
                    throw new ArgumentException("The labels length " +
                        "doesn't correspond to the initialized instances count.");
            }
            else InstancesCount = labels.Length;

            for (var i = 0; i < labels.Length; i++)
            {
                if (categoriesMap.ContainsKey(labels[i]))
                {
                    CategoricalLabels[i] = categoriesMap[labels[i]];
                }
                else
                {
                    categoriesMap.Add(labels[i], categoriesMap.Count);
                    CategoricalLabels[i] = categoriesMap[labels[i]];
                    TrueLabelsMap.Add(categoriesMap[labels[i]], labels[i]);
                }
            }
        }

        public void SortSparseInput(
            float[] values,
            out float[] nonZeroSortedValues,
            out int[] nonZeroSortedIndices,
            out int negCount)
        {

            var sortedIndices = new int[values.Length];
            var sortedValues = new float[values.Length];
            for (var i = 0; i < values.Length; i++)
            {
                sortedIndices[i] = i;
            }

            Array.Copy(values, sortedValues, values.Length);
            Array.Sort(sortedValues, sortedIndices);

            negCount = 0;
            for (var i = 0; i < sortedValues.Length; i++)
            {
                if (sortedValues[i] >= 0f)
                {
                    break;
                }

                negCount++;
            }

            var zeroCount = 0;
            for (var i = negCount; i < sortedValues.Length; i++)
            {
                if (sortedValues[i] > 0f)
                {
                    break;
                }

                zeroCount++;
            }

            nonZeroSortedIndices = new int[sortedValues.Length - zeroCount];
            nonZeroSortedValues = new float[sortedValues.Length - zeroCount];

            // copy the negative values
            if(negCount >= 0)// If we have negative values.
            {
                Array.Copy(sortedValues, 0, nonZeroSortedValues, 0, negCount);
                Array.Copy(sortedIndices, 0, nonZeroSortedIndices, 0, negCount);
            }

            // copy the positive values
            Array.Copy(sortedValues, negCount + zeroCount, nonZeroSortedValues, negCount,
                nonZeroSortedValues.Length - negCount);
            Array.Copy(sortedIndices, negCount + zeroCount, nonZeroSortedIndices, negCount,
                nonZeroSortedValues.Length - negCount);
        }

        /// <summary>
        /// Draws random subset of instances and labels. Here we create
        /// new objects for the values and their relative mapping to the
        /// labels so that we can freely reset their scopeOffset and -Count.
        /// </summary>
        /// <param name="noReplacement"> 
        /// It indicates the whether we want sampling with
        /// or without replacement. The one without replacement
        /// is way more computationaly intensice.
        /// </param>
        public (int[], List<IFeature>) DrawRandomInstances(
            int subsetSize,
            Random random,
            bool noReplacement = true,
            bool keepAllLabels = true)
        {
            // If we have any instances we get their count.
            var sampleCount = InstancesCount;

            var indices = Enumerable.Range(0, sampleCount).ToArray();
            var subsetIndices = noReplacement
                ? indices.SampleNoReplacement(subsetSize, random)
                : indices.SampleReplacement(subsetSize, random);

            var subsampleOccurences = new int[sampleCount];
            for(var i = 0; i < subsetIndices.Length; i++)
            {
                subsampleOccurences[subsetIndices[i]] += 1;
            }

            var subsetLabels = new int[subsetSize];
            var subsampleIndexMapper = new Dictionary<int,int>();
            for (int i = 0, k = 0; i < subsampleOccurences.Length; i++)
            {
                var to = subsampleOccurences[i];

                for (var j = 0; j < to; j++)
                {
                    subsampleIndexMapper[i] = k;

                    subsetLabels[k] = CategoricalLabels[i];
                    k++;
                }
            }

            var subsetFeatures = new List<IFeature>();
            for (int i = 0; i < Features.Count; i++)
            {
                subsetFeatures.Add(Features[i].SubsampleFeature(subsampleOccurences, subsampleIndexMapper, subsetSize));
            }

            return (subsetLabels, subsetFeatures);
        }

        /// <summary>
        /// Draws random subset of feature set.
        /// </summary>
        /// <param name="noReplacement"> 
        /// It indicates the whether we want sampling with
        /// or without replacement. The one without replacement
        /// is way more computationaly intensice.
        /// </param>
        public FeatureRepresentation DrawRandomSubset(int subsetSize, Random random, bool noReplacement = true)
        {
            (var subsetLabels, var subsetFeatures) = DrawRandomInstances(subsetSize, random, noReplacement);
            return With(
                instancesCount: subsetSize,
                features: subsetFeatures,
                categoricalLabels: subsetLabels);
        }

        public (FeatureRepresentation, FeatureRepresentation) SplitFeatures(
            bool[] isLabelOutOfScopeLeft,
            bool[] isLabelOutOfScopeRight)
        {
            var leftSplitFeatures = new List<IFeature>();
            var rightSplitFeatures = new List<IFeature>();

            for (int i = 0; i < Features.Count; i++)
            {
                var (leftSplit, rightSplit) = Features[i].Split(
                    isLabelOutOfScopeLeft: isLabelOutOfScopeLeft,
                    isLabelOutOfScopeRight: isLabelOutOfScopeRight);

                leftSplitFeatures.Add(leftSplit);
                rightSplitFeatures.Add(rightSplit);
            }

            return (With(features: leftSplitFeatures), With(features: rightSplitFeatures));
        }

        public FeatureRepresentation With(
            int? instancesCount = null,
            List<IFeature> features = null,
            List<InputFeatureTypes> featureTypes = null,
            Dictionary<int, float> trueLabelsMap = null,
            int[] categoricalLabels = null,
            bool? isSortedDataset = null)
        {
            return new FeatureRepresentation(
            instancesCount: instancesCount ?? InstancesCount,
            features: features ?? Features,
            featureTypes: featureTypes ?? FeatureTypes,
            trueLabelsMap: trueLabelsMap ?? TrueLabelsMap,
            categoricalLabels: categoricalLabels ?? CategoricalLabels,
            isSortedDataset: isSortedDataset ?? IsSortedDataset);
        }
    }
}
