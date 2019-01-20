
using ML.MathHelpers;
using ML.RandomForest;
using System;
using System.Collections.Generic;

namespace ML
{
    /// <summary>
    /// The actual implementation of the Sparse and Dense Features.
    /// Each implementations has particular algorithms taking advantage
    /// of the sparse and dense structure of the instances.
    /// </summary>
    public class Features
    {
        public class DenseFeature: IFeature
        {
            /// <summary>
            /// Percision coefficient by deciding the split.
            /// </summary>
            private const float Epsilon = 1e-16f;
            private float[] _values;
            /// <summary>
            /// This is used just in case of sorted features.
            /// </summary>
            private int[] _sortedIndices;

            /// <summary>
            /// The total count of samples within a feature.
            /// </summary>
            public int Length => _values.Length;

            /// <summary>
            /// Donates the index of the first index of the instance values.
            /// It is used in the cases where we mutate the feature focusing just
            /// on particular scope of the instaces within it.
            /// </summary>
            private int _scopeOffset;
            /// <summary>
            /// Donates the count of the instances contained in the feature
            /// in the cases where we mutate the feature focusing just on
            /// particular scope of the instaces within it.
            /// </summary>
            private int _scopeCount;

            public bool IsRedundant { get; }

            public DenseFeature(float[] values)
            {
                _values = values;
                IsRedundant =
                    values.Length < 2 ||
                    values.IsHomogenious(0, values.Length);
            }

            public DenseFeature(float[] values, int[] sortedIndices, int scopeOffset, int scopeCount)
            {
                _values = values;
                _scopeOffset = scopeOffset;
                _scopeCount = scopeCount;
                _sortedIndices = sortedIndices;
                IsRedundant = 
                    values.Length < 2 ||
                    scopeCount < 2 ||
                    values.IsHomogenious(_scopeOffset, _scopeCount) ||
                    scopeOffset > values.Length - 3;// We need at least two values for a split.
            }

            public void FindBestSplit(
                IReadOnlyList<int> labels,
                int[] labelsCounts,
                int labelsCount,
                IDecisionMetric<int> metric,
                IReadOnlyList<bool> initialIsLabelOutOfScope,
                out bool[] isLabelOutOfScopeLeft,
                out bool[] isLabelOutOfScopeRight,
                out float threshold,
                out double minCost)
            {
                minCost = double.MaxValue;
                var bestFirstRightIndex = _scopeOffset;

                var partialCounts = new int[labelsCounts.Length];

                var scopeLimit = _scopeOffset + _scopeCount - 1;
                for (var i = _scopeOffset; i < scopeLimit; i++)
                {
                    partialCounts[labels[_sortedIndices[i]]]++;

                    if (_values[i + 1] - _values[i] < Epsilon)
                    {
                        continue;
                    }

                    var cost = metric.Calculate(labelsCounts, partialCounts, labelsCount);

                    if (minCost > cost)
                    {
                        minCost = cost;
                        bestFirstRightIndex = i + 1;
                    }
                }

                threshold = _values[bestFirstRightIndex];
                isLabelOutOfScopeLeft = new bool[initialIsLabelOutOfScope.Count];
                isLabelOutOfScopeRight = new bool[initialIsLabelOutOfScope.Count];

                for (var i = 0; i < initialIsLabelOutOfScope.Count; i++)
                {
                    isLabelOutOfScopeLeft[i] = true;
                    isLabelOutOfScopeRight[i] = true;
                }

                for (var i = _scopeOffset; i < bestFirstRightIndex; i++)
                {
                    isLabelOutOfScopeLeft[_sortedIndices[i]] = false;
                }

                scopeLimit = _scopeOffset + _scopeCount;

                for (var i = bestFirstRightIndex; i < scopeLimit; i++)
                {
                    isLabelOutOfScopeRight[_sortedIndices[i]] = false;
                }
            }

            /// <summary> Subsamples instances from the feature. </summary>
            /// <param name="occurrences">
            /// The time each instance is appearing in the subsample.
            /// </param>
            /// <param name="indexMapper">
            /// Maps the dataset indices to the subset indices.
            /// </param>
            public IFeature SubsampleFeature(
                IReadOnlyList<int> occurrences,
                Dictionary<int, int> indexMapper,
                int subsampleCount)
            {
                var subsampledValues = new float[subsampleCount];

                var scopeLimit = _scopeOffset + _scopeCount;

                if (_sortedIndices != null)
                {
                    var subsampledIndices = new int[subsampleCount];

                    for (int i = _scopeOffset, k = 0; i < scopeLimit; i++)
                    {
                        var v = _values[i];
                        var idx = _sortedIndices[i];

                        var to = occurrences[idx];

                        for (var j = 0; j < to; j++)
                        {
                            subsampledValues[k] = v;
                            subsampledIndices[k] = indexMapper[idx];
                            k++;
                        }
                    }
                    return new DenseFeature(
                        values: subsampledValues, 
                        sortedIndices: subsampledIndices, 
                        scopeOffset: 0, 
                        scopeCount: subsampledIndices.Length);
                }

                for (int i = _scopeOffset, k = 0; i < scopeLimit; i++)
                {
                    var v = _values[i];

                    var to = occurrences[i];

                    for (var j = 0; j < to; j++)
                    {
                        subsampledValues[k++] = v;
                    }
                }

                return new DenseFeature(subsampledValues);
            }

            public (IFeature, IFeature) Split(
                IReadOnlyList<bool> isLabelOutOfScopeLeft,
                IReadOnlyList<bool> isLabelOutOfScopeRight)
            {
                var leftSplitIndices = new List<int>();
                var rightSplitIndices = new List<int>();

                var leftSplitValues = new List<float>();
                var rightSplitValues = new List<float>();

                var scopeLimit = _scopeOffset + _scopeCount;

                for (var i = _scopeOffset; i < scopeLimit; i++)
                {
                    if (isLabelOutOfScopeLeft[_sortedIndices[i]])
                    {
                        rightSplitIndices.Add(_sortedIndices[i]);
                        rightSplitValues.Add(_values[i]);
                    }
                    else
                    {
                        leftSplitIndices.Add(_sortedIndices[i]);
                        leftSplitValues.Add(_values[i]);
                    }
                }

                var leftOffset = _scopeOffset;

                for (var i = 0; i < leftSplitIndices.Count; i++)
                {
                    _sortedIndices[_scopeOffset + i] = leftSplitIndices[i];
                    _values[_scopeOffset + i] = leftSplitValues[i];
                }

                var rightOffset = _scopeOffset + leftSplitIndices.Count;

                for (var i = 0; i < rightSplitIndices.Count; i++)
                {
                    _sortedIndices[rightOffset + i] = rightSplitIndices[i];
                    _values[rightOffset + i] = rightSplitValues[i];
                }

                var leftSplit = new DenseFeature(
                    values: _values,
                    sortedIndices: _sortedIndices,
                    scopeOffset: leftOffset,
                    scopeCount: leftSplitIndices.Count);

                var rightSplit = new DenseFeature(
                    values: _values,
                    sortedIndices: _sortedIndices,
                    scopeOffset: rightOffset,
                    scopeCount: rightSplitIndices.Count);

                return (leftSplit, rightSplit);
            }

            public float[] GetValues()
            {
                var values = new float[Length];

                if(_sortedIndices != null) // If the feature is sorted.
                {
                    for (var i = 0; i < _sortedIndices.Length; i++)
                    {
                        values[_sortedIndices[i]] = _values[i];
                    }
                    return values;
                }

                for (var i = 0; i < _values.Length; i++)
                {
                    values[i] = _values[i];
                }

                return values;
            }
        }

        public class SparseFeature : IFeature
        {
            /// <summary>
            /// Percision coefficient by deciding the split.
            /// </summary>
            private const float Epsilon = 1e-16f;

            private float[] _values;
            private int[] _sortedIndices;

            /// <summary>
            /// The total count of samples within a feature. Including positive
            /// negative and zero values.
            /// </summary>
            public int Length { get; }

            /// <summary>
            /// Donates the index of the first index of the instance values.
            /// It is used in the cases where we mutate the feature focusing just
            /// on particular scope of the instaces within it.
            /// </summary>
            private int _scopeOffset;
            /// <summary>
            /// Donates the count of the instances contained in the feature
            /// in the cases where we mutate the feature focusing just on
            /// particular scope of the instaces within it. It consists of
            /// the count of negative and the count of the positive values in
            /// scope.
            /// </summary>
            private int _scopeCount;

            /// <summary>
            /// Represents the count of the negative values.
            /// Needed in the case, where we have sorted
            /// values, and we have to preserve information about the
            /// boundries of the zero values.
            /// </summary>
            private readonly int _negCount;

            public bool IsRedundant { get; }

            public SparseFeature(
                float[] values,
                int[] sortedIndices,
                int length,
                int scopeOffset,
                int scopeCount,
                int negCount)
            {
                _values = values;
                _sortedIndices = sortedIndices;
                _scopeOffset = scopeOffset;
                _scopeCount = scopeCount;
                _negCount = negCount;
                Length = length;

                IsRedundant =
                    length < 2 ||
                    values.IsHomogenious(_scopeOffset, _scopeCount) ||
                    scopeCount < 1 ||
                    scopeOffset > sortedIndices.Length - 1;
            }

            /// <summary> Subsamples instances from the feature. </summary>
            /// <param name="occurrences">
            /// The time each instance is appearing in the subsample.
            /// </param>
            /// <param name="indexMapper">
            /// Maps the dataset indices to the subset indices.
            /// </param>
            public IFeature SubsampleFeature(
                IReadOnlyList<int> occurrences,
                Dictionary<int, int> indexMapper,
                int subsampleCount)
            {
                var count = 0;
                // If the feature is not sorted than negCount is -1.
                var negCount = Math.Min(_negCount, 0);

                var negEnd = Math.Min(_negCount, _scopeOffset + _scopeCount);

                for (var i = _scopeOffset; i < negEnd; i++)
                {
                    if (occurrences[_sortedIndices[i]] > 0)
                    {
                        negCount += occurrences[_sortedIndices[i]];
                    }
                }

                count = Math.Max(negCount, 0);
                var positOffset = Math.Max(negEnd, 0);

                var scopeLimit = _scopeOffset + _scopeCount;

                for (int i = positOffset; i < scopeLimit; i++)
                {
                    count += occurrences[_sortedIndices[i]];
                }

                var subsampledValues = new float[count];
                var subsampledIndices = new int[count];

                for (int i = _scopeOffset, k = 0; i < scopeLimit; i++)
                {
                    var v = _values[i];
                    var idx = _sortedIndices[i];

                    var to = occurrences[idx];

                    for (var j = 0; j < to; j++)
                    {
                        subsampledValues[k] = v;
                        subsampledIndices[k] = indexMapper[idx];
                        k++;
                    }
                }

                return new SparseFeature(
                    values: subsampledValues,
                    sortedIndices: subsampledIndices,
                    length: subsampleCount,
                    // We reset the scopeOffset and scopeCount.
                    scopeOffset: 0,
                    scopeCount: count,
                    negCount: negCount);
            }

            public float[] GetValues()
            {
                var values = new float[Length];

                for(var i = 0; i < _sortedIndices.Length; i++)
                {
                    values[_sortedIndices[i]] = _values[i];
                }

                return values;
            }
        }

        public class BinaryFeature : IFeature
        {
            /// <summary>
            /// Represent the indices in the one values.
            /// </summary>
            private int[] _sortedIndices;

            /// <summary>
            /// The total count of samples within a feature/
            /// </summary>
            public int Length { get; }

            /// <summary>
            /// Donates the index of the first index of the instance values.
            /// It is used in the cases where we mutate the feature focusing just
            /// on particular scope of the instaces within it.
            /// </summary>
            private int _scopeOffset;
            /// <summary>
            /// Donates the count of the instances contained in the feature
            /// in the cases where we mutate the feature focusing just on
            /// particular scope of the instaces within it.
            /// </summary>
            private int _scopeCount;

            public bool IsRedundant { get; }

            public BinaryFeature(
                int[] sortedIndices,
                int length,
                int scopeOffset,
                int scopeCount)
            {
                _sortedIndices = sortedIndices;
                Length = length;
                _scopeOffset = scopeOffset;
                _scopeCount = scopeCount;
                IsRedundant = 
                    length < 2 ||
                    scopeCount < 1 ||
                    scopeCount == length ||
                    scopeCount == 0 ||
                    scopeOffset > sortedIndices.Length - 1;
            }

            /// <summary> Subsamples instances from the feature. </summary>
            /// <param name="occurrences">
            /// The time each instance is appearing in the subsample.
            /// </param>
            /// <param name="indexMapper">
            /// Maps the dataset indices to the subset indices.
            /// </param>
            public IFeature SubsampleFeature(
                IReadOnlyList<int> occurrences,
                Dictionary<int, int> indexMapper,
                int subsampleCount)
            {
                var count = 0;

                var scopeLimit = _scopeOffset + _scopeCount;

                for (int i = _scopeOffset; i < scopeLimit; i++)
                {
                    count += occurrences[_sortedIndices[i]];
                }

                var subsampledIndices = new int[count];
                for (int i = _scopeOffset, k = 0; i < scopeLimit; i++)
                {
                    var idx = _sortedIndices[i];
                    var to = occurrences[idx];

                    for (var j = 0; j < to; j++)
                    {
                        subsampledIndices[k++] = indexMapper[idx];
                    }
                }

                return new BinaryFeature(
                    sortedIndices: subsampledIndices,
                    length: subsampleCount,
                    // We reset the scopeOffset and scopeCount.
                    scopeOffset: 0,
                    scopeCount: count);
            }

            public float[] GetValues()
            {
                var values = new float[Length];

                for (var i = 0; i < _sortedIndices.Length; i++)
                {
                    values[_sortedIndices[i]] = 1f;
                }

                return values;
            }
        }
    }
}
