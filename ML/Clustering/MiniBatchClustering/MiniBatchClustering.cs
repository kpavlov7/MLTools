using System;
using ML.MathHelpers;

namespace ML.Clustering
{
    /// <summary>
    /// Algorithm for fast k-means clustering. The alorithm also accommodates sparse representation of the
    /// feature space: www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf
    /// </summary>
    public class MiniBatchClustering
    {
        /// <summary>
        /// Cluster count;
        /// </summary>
        public int K { get; set; }

        private int _minibatchSize { get; set; }

        private int _iterationsCount { get; set; }

        /// <summary>
        /// Random generator seed;
        /// </summary>
        private const int Seed = 12;

        /// <summary>
        /// Tolerance level;
        /// </summary>
        private const double Epsilon = 0.001;

        /// <summary>
        /// Ball radius;
        /// </summary>
        private const double Lambda = 1;

        private Random _random;

        private int FeaturesCount { get; set; }

        /// <summary>
        /// Indicator whether we should use k-means++ initialization.
        /// If false we use random initialization.
        /// </summary>
        private bool _usePlusPlusInit;

        /// <summary>
        /// Indicates whether we've already started the training in
        /// case of mini batch updates.
        /// </summary>
        private bool _isInitialized;

        private float[,] _centroids;

        public MiniBatchClustering(
            int k,
            int minibatchSize,
            int iterationsCount,
            bool sparse = false,
            bool usePlusPlusInit = true)
        {
            _usePlusPlusInit = usePlusPlusInit;
            K = k;
            _minibatchSize = minibatchSize;
            _iterationsCount = iterationsCount;
            _random = new Random(Seed);
            _centroids = new float[0, 0];
        }

        /// <summary>
        /// Trains the centroids.
        /// </summary>
        public void Train(InstanceRepresentation set)
        {
            var instances = set.Instances.ToArray();

            // Initializing the centroids:
            if (_usePlusPlusInit)
            {
                _centroids = PlusPlusInitializer.InitializeCentroids(K, instances, _random);
            }
            else
            {
                _centroids = Instances.ConvertToArray(instances.SampleNoReplacement(K, _random));
            }

            _isInitialized = true;
            FeaturesCount = set.FeauturesCount;

            for (var i = 0; i < _iterationsCount; i++)
            {
                var miniBatch = instances.SampleReplacement(_minibatchSize, _random);
                MiniBatchUpdate(miniBatch, set.IsSparseDataset);
            }
        }

        public void MiniBatchUpdate(IInstance[] miniBatch, bool isSparseMiniBatch)
        {
            var minibathSize = miniBatch.Length;
            if (!_isInitialized)
            {
                // Initializing the centroids:
                if (_usePlusPlusInit)
                {
                    _centroids = PlusPlusInitializer.InitializeCentroids(K, miniBatch, _random);
                }
                else
                {
                    _centroids = Instances.ConvertToArray(
                        miniBatch.SampleReplacement(FeaturesCount, _random));
                }

                _isInitialized = true;
            }

            var nearestClusters = new int[_minibatchSize];
            var perCenterCount = new int[_centroids.Length];

            for (var j = 0; j < minibathSize; j++)
            {
                nearestClusters[j] = Instances.MinEucDistanceIndex(miniBatch[j], _centroids);
            }

            // If the dataset is not sparse we perform the non sparse cluster computation.
            if (!isSparseMiniBatch)
            {
                for (var j = 0; j < minibathSize; j++)
                {
                    perCenterCount[nearestClusters[j]] += 1;
                    var learningRate = 1.0 / perCenterCount[nearestClusters[j]];

                    for (var k = 0; k < FeaturesCount; k++)
                    {
                        var c = _centroids[nearestClusters[j], k];
                        _centroids[nearestClusters[j], k] = (float)((1.0 - learningRate) * c + miniBatch[j].GetValue(k) * learningRate);
                    }
                }
            }
            else // If the dataset is sparse we perform the sparse clustering version.
            {
                for (var j = 0; j < minibathSize; j++)
                {
                    var current = _centroids.L1Norm(nearestClusters[j], FeaturesCount);

                    if (current <= Epsilon + Lambda)
                    {
                        break;
                    }

                    var upper = _centroids.Max(nearestClusters[j], FeaturesCount);
                    var lower = 0.0;
                    var theta = 0.0;

                    while (current < Lambda * (Epsilon + 1) || current < Lambda)
                    {
                        theta = (upper + lower) / 2.0; // Get L1 value
                        current = 0.0;
                        for (var k = 0; k < FeaturesCount; k++)
                        {
                            current += Math.Max(0, Math.Abs(_centroids[nearestClusters[j], k]) - theta);
                            if (current <= Lambda)
                            {
                                upper = theta;
                            }
                            else
                            {
                                lower = theta;
                            }
                        }
                    }

                    for (var k = 0; k < FeaturesCount; k++)
                    {
                        var c = _centroids[nearestClusters[j], k];
                        _centroids[nearestClusters[j], k] = (float)(Math.Sign(c) * Math.Max(0, Math.Abs(c) - theta));
                    }
                }
            }
        }

        /// <summary>
        /// Directly train and cluster the provided dataset.
        /// </summary>
        public int[] Cluster(InstanceRepresentation set)
        {
            if (!_isInitialized)
            {
                throw new InvalidOperationException("The centroids are not trained.");
            }

            var instancesClusters = new int[set.Instances.Count];
            for (var i = 0; i < set.Instances.Count; i++)
            {
                instancesClusters[i] = Instances.MinEucDistanceIndex(set.Instances[i], _centroids);
            }

            return instancesClusters;
        }
    }
}
