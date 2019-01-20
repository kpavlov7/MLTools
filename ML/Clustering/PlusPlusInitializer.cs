using ML.MathHelpers;
using System;
using System.Collections.Generic;

namespace ML.Clustering
{
    /// <summary>
    ///  The k-means++ algorithm for smart initializing of centroids.
    ///  https://en.wikipedia.org/wiki/K-means%2B%2B
    /// </summary>
    public static class PlusPlusInitializer
    {
        /// <summary>
        /// Initializes centroids.
        /// </summary>
        public static float[,] InitializeCentroids(int kn, IReadOnlyList<IInstance> instances, Random random)
        {
            var featuresCount = instances[0].Length;
            var instancesCount = instances.Count;

            if (kn >= instancesCount)
            {
                throw new ArgumentException(
                    "The instances count should be bigger than the centroids count.");
            }

                float[,] means = new float[kn, featuresCount];
            List<int> used = new List<int>();

            int idx = random.Next(0, featuresCount);
            means.FillTensor(0, 0, 0, featuresCount, instances[idx].GetValues());
            used.Add(idx);

            for (int k = 1; k < kn; ++k)
            {
                var dSquared = new double[instancesCount];
                var newMean = -1;

                for (int i = 0; i < instancesCount; ++i)
                {
                    if (used.Contains(i) == true)
                    {
                        continue;
                    }

                    var distances = new double[k];

                    for (int j = 0; j < k; ++j)
                    {
                        distances[j] = instances[i].L2Dist(means, j, featuresCount);
                    }

                    int m = distances.MinArg();
                    dSquared[i] = distances[m] * distances[m];
                }

                var p = random.NextDouble();
                var sum = 0.0;

                for (var i = 0; i < dSquared.Length; ++i)
                {
                    sum += dSquared[i];
                }

                var cumulative = 0.0;
                var ii = 0;
                var sanity = 0;

                while (sanity < instancesCount * 2)
                {
                    cumulative += dSquared[ii] / sum;
                    if (cumulative >= p && used.Contains(ii) == false)
                    {
                        newMean = ii; // the chosen index
                        used.Add(newMean); // don't pick again
                        break;
                    }

                    ++ii; // next candidate

                    if (ii >= dSquared.Length)
                    {
                        ii = 0;
                    } // past the end

                    ++sanity;
                }

                means.FillTensor(k, 0, 0, featuresCount, instances[newMean].GetValues());
            } // k, each remaining mean

            return means;
        }
    }
}
