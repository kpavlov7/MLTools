using System;
using ML.MathHelpers;

namespace ML
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
        private int _k { get; set; }

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

        private Random _random  { get; set; }

        private int _featuresCount;

        public MiniBatchClustering(int k, int minibatchSize, int iterationsCount, bool sparse = false)
        {
            _k = k;
            _minibatchSize = minibatchSize;
            _iterationsCount = iterationsCount;
            _random = new Random(Seed);
        }

        //TODO: separate the 
        //TODO: implement method which only Updates the cluster centroids.
        /// <summary>
        /// Directly train and cluster the provided dataset.
        /// </summary>
        public int[] Cluster(InstanceRepresentation set)
        {
            // Initializing the centroids:
            var centroids = set.DrawRandomSubsetValues(_k, _random);
            var perCenterCount = new int[centroids.Length];
            _featuresCount = set.FeauturesCount;

            for (var i = 0; i < _iterationsCount; i++)
            {
                // TODO: implement more cache friendly approach without storing the whole minibatch.
                var miniBatch = set.DrawRandomSubset(_minibatchSize, _random);
                var nearestClusters = new int[_minibatchSize];

                for (var j = 0; j < miniBatch.Length; j++)
                {
                    nearestClusters[j] = set.MinEucDistanceIndex(miniBatch[j], centroids);
                }

                // If the dataset is not sparse we perform the non sparse cluster computation
                if (!set.IsSparseDataset)
                {
                    for (var j = 0; j < miniBatch.Length; j++)
                    {
                        perCenterCount[nearestClusters[j]] += 1;
                        var learningRate = 1.0 / perCenterCount[nearestClusters[j]];

                        for (var k = 0; k < _featuresCount; k++)
                        {
                            var c = centroids[nearestClusters[j], k];
                            centroids[nearestClusters[j], k] = (float)((1.0 - learningRate) * c + miniBatch[j].GetValue(k) * learningRate);
                        }
                    }
                }
                else // If the dataset is sparse we perform the sparse clustering version
                {
                    for (var j = 0; j < miniBatch.Length; j++)
                    {
                        var current = centroids.L1Norm(nearestClusters[j], _featuresCount);

                        if (current <= Epsilon + Lambda)
                        {
                            break;
                        }

                        var upper = centroids.Max(nearestClusters[j], _featuresCount);
                        var lower = 0.0;
                        var theta = 0.0;

                        while (current < Lambda * (Epsilon + 1) || current < Lambda)
                        {
                            theta = (upper + lower) / 2.0; // Get L1 value
                            current = 0.0;
                            for (var k = 0; k < _featuresCount; k++)
                            {
                                current += Math.Max(0, Math.Abs(centroids[nearestClusters[j], k]) - theta);
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

                        for (var k = 0; k < _featuresCount; k++)
                        {
                            var c = centroids[nearestClusters[j], k];
                            centroids[nearestClusters[j], k] = (float)(Math.Sign(c) * Math.Max(0, Math.Abs(c) - theta));
                        }
                    }
                }
            }
            var instancesClusters = new int[set.Instances.Count];
            for (var i = 0; i < set.Instances.Count; i++)
            {
                instancesClusters[i] = set.MinEucDistanceIndex(set.Instances[i], centroids);
            }

            return instancesClusters;
        }
    }
}
