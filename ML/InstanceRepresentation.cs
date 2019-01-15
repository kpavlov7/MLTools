using System.Collections.Generic;
using System;
using System.Linq;
using static ML.Instances;
using ML.MathHelpers;

namespace ML
{
    /// <summary>
    /// The Instance Representation is intended as a dataset implementation in suitable
    /// format for Machine Learning.
    /// </summary>
    public class InstanceRepresentation
    {
        /// <summary>
        /// Total features count;
        /// </summary>
        public int FeauturesCount { get; set; }

        private int[] _ordinalMapping;
        private int[] _flagsMapping;
        private int[] _featureMapper;

        public List<IInstance> Instances { get; set; }

        public bool IsSparseDataset;

        public InstanceRepresentation(
            InputFeatureTypes[] featureTypes, bool sparse = false)
        {
            Instances = new List<IInstance>();
            FeauturesCount = featureTypes.Length;
            var ordinalIndices = new List<int>();
            var flagsIndices = new List<int>();

            for (var i = 0; i < featureTypes.Length; i++)
            {
                if (featureTypes[i] == InputFeatureTypes.Ordinal)
                {
                    ordinalIndices.Add(i);
                }

                if (featureTypes[i] == InputFeatureTypes.Flags)
                {
                    flagsIndices.Add(i);
                }
            }

            _ordinalMapping = ordinalIndices.ToArray();
            _flagsMapping = flagsIndices.ToArray();

            _featureMapper = new int[_ordinalMapping.Length + _flagsMapping.Length];
            for(var i = 0; i < _ordinalMapping.Length; i++)
            {
                _featureMapper[_ordinalMapping[i]] = i;
            }

            for (var i = 0; i < _flagsMapping.Length; i++)
            {
                _featureMapper[_flagsMapping[i]] = i + _ordinalMapping.Length;
            }

            IsSparseDataset = sparse;
        }

        public void AddInstance(float[] input)
        {
            if (IsSparseDataset)
            {
                var values = new List<float>();
                var indices = new List<int>();
                var binaryIndices = new List<int>();

                for (int i = 0; i < _flagsMapping.Length; i++)
                {
                    var bv = input[_flagsMapping[i]];

                    if (bv != 1f && bv != 0f)
                    {
                        throw new ArgumentException("The binary values should be either 1 or 0.");
                    }
                    else
                    {
                        if (bv != 0)
                        {
                            binaryIndices.Add(i);
                        }
                    }
                }

                for (int i = 0; i < _ordinalMapping.Length; i++)
                {
                    var ov = input[_ordinalMapping[i]];
                    if (ov != 0)
                    {
                        indices.Add(i);
                        values.Add(ov);
                    }
                }
                Instances.Add(new SparseInstance(
                    values: values.ToArray(),
                    indices: indices.ToArray(),
                    binaryIndices: binaryIndices.ToArray(),
                    valuesLength: _ordinalMapping.Length,
                    binaryLength: _flagsMapping.Length));
            }
            else
            {
                var orderedValues = new float[FeauturesCount];

                for (int i = 0; i < _ordinalMapping.Length; i++)
                {
                    orderedValues[i] = input[_ordinalMapping[i]];
                }

                for (int i = 0; i < _flagsMapping.Length; i++)
                {
                    orderedValues[i + _ordinalMapping.Length] = input[_flagsMapping[i]];
                }

                Instances.Add(new DenseInstance(orderedValues, _ordinalMapping.Length));
            }
        }

        public float GetValue(int instanceIndex, int featureIndex)
        {
            return Instances[instanceIndex].GetValue(_featureMapper[featureIndex]);
        }

        public IInstance DrawRandomInstance(Random random)
        {
            var r = random.Next(Instances.Count);

            return Instances[r].Copy();
        }

        /// <summary>
        /// Draws random subset of instances.
        /// </summary>
        /// <param name="noReplacement"> 
        /// It indicates the whether we want sampling with
        /// or without replacement. The one without replacement
        /// is way more computationaly intensice.
        /// </param>
        public IInstance[] DrawRandomSubset(int subsetSize, Random random, bool noReplacement = true)
        {
            var subset = noReplacement
                ? Instances.ToArray().SampleNoReplacement(subsetSize, random)
                : Instances.ToArray().SampleReplacement(subsetSize, random);

            return subset;
        }

        /// <summary>
        /// Draws random subset of instance values.
        /// </summary>
        /// <param name="noReplacement"> 
        /// It indicates the whether we want sampling with
        /// or without replacement. The one without replacement
        /// is way more computationaly intensice.
        /// </param>
        public float[,] DrawRandomSubsetValues(int subsetSize, Random random, bool noReplacement = true)
        {
            var instanceSubset = noReplacement
                ? Instances.ToArray().SampleNoReplacement(subsetSize, random)
                : Instances.ToArray().SampleReplacement(subsetSize, random);

            var values = new float[subsetSize, FeauturesCount];

            for (var i = 0; i < subsetSize; i++)
            {
                var instanceValues = instanceSubset[i].GetValues();
                for (var j = 0; j < FeauturesCount; j++)
                {
                    values[i, j] = instanceValues[j];
                }
            }
            return values;
        }

        public int MinEucDistanceIndex(IInstance targetInstance, IInstance[] instances)
        {
            var minDist = double.MaxValue;
            var minDistIndex = 0;

            for (var i = 0; i < instances.Length; i++)
            {
                var dist = targetInstance.L2Dist(instances[i]);
                if (minDist > dist)
                {
                    minDist = dist;
                    minDistIndex = i;
                }
            }
            return minDistIndex;
        }

        public int MinEucDistanceIndex(IInstance targetInstance, float[,] values)
        {
            var minDist = double.MaxValue;
            var minDistIndex = 0;

            var valuesCount = values.GetLength(0);
            var valuesDimCount = values.GetLength(1);

            for (var i = 0; i < valuesCount; i++)
            {
                var dist = targetInstance.L2Dist(values, i, valuesDimCount);
                if (minDist > dist)
                {
                    minDist = dist;
                    minDistIndex = i;
                }
            }
            return minDistIndex;
        }

        public void Rescale ()
        {
            var max = new float[_ordinalMapping.Length];
            max.Repeat(float.MinValue);

            var min = new float[_ordinalMapping.Length];
            min.Repeat(float.MaxValue);

            for (var i = 0; i < Instances.Count; i++)
            {
                var values = Instances[i].GetOrdinals();

                for (var j = 0; j < _ordinalMapping.Length; j++)
                {
                    if (max[j] < values[j])
                    {
                        max[j] = values[j];
                    }
                    if (min[j] > values[j])
                    {
                        min[j] = values[j];
                    }
                }
            }

            for (var i = 0; i < Instances.Count; i++)
            {
                Instances[i].Rescale(min, max);
            }
        }

        public void Standardize()
        {
            var sum = new double[_ordinalMapping.Length]; //square sum for estimating sigma and mean
            var ssum = new double[_ordinalMapping.Length]; //square sum for estimating sigma
            var mean = new float[_ordinalMapping.Length];

            for (var i = 0; i < Instances.Count; i++)
            {
                var values = Instances[i].GetOrdinals();

                for (var j = 0; j < values.Length; j++)
                {
                    sum[j] += values[j];
                    ssum[j] += values[j] * values[j];
                }
            }

            var sigma = new float[_ordinalMapping.Length];
            for (var i = 0; i < _ordinalMapping.Length; i++)
            {
                mean[i] = (float)sum[i] / Instances.Count;
                sigma[i] = (float)Math.Sqrt(ssum[i] / Instances.Count - mean[i] * mean[i]);
            }

            for (var i = 0; i < Instances.Count; i++)
            {
                Instances[i].Standardize(mean, sigma);
            }

            // In case of sparse data we remove the binary mapping, because the binary values are standardized.
            if (IsSparseDataset)
            {
                var newOrdinalMapping = new int[_ordinalMapping.Length + _flagsMapping.Length];
                Array.Copy(_ordinalMapping, newOrdinalMapping, _ordinalMapping.Length);
                Array.Copy(_flagsMapping, 0, newOrdinalMapping, _ordinalMapping.Length, _flagsMapping.Length);
                _flagsMapping = new int[0];
            }
        }
    }
}
