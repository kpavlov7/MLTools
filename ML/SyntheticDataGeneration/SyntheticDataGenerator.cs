using System;
using System.Collections.Generic;
using ML.MathHelpers;

namespace ML
{
    /// <summary>
    /// According to the master thesis 'SYNTHETIC DATASETS FOR CLUSTERING ALGORITHMS' by Jhansi Rani Vennam
    /// https://researchweb.iiit.ac.in/~jhansi/Thesis.pdf
    /// </summary>
    public class SyntheticDataGenerator
    {
        /// <summary>
        /// The value cape for the whole feature space;
        /// </summary>
        private const double ValueCap = 300;

        /// <summary>
        /// Maximum allowed radius;
        /// </summary>
        private double _maxRadius { get; set; }

        /// <summary>
        /// Minimum allowed radius;
        /// </summary>
        private double _minRadius { get; set; }

        // Random generator initializaton
        private const int RandomSeed = 12;
        private static Random Random = new Random(RandomSeed);

        /// <summary>
        /// Allowed ratio between max radius and min radius;
        /// </summary>
        private const int RR = 3;

        /// <summary>
        /// Total count of observations;
        /// </summary>
        private readonly int N;

        /// <summary>
        /// Total count of noise points; The noise points are scattered in
        /// the feature space, but outside of clusters space. 
        /// </summary>
        private readonly int P;
        /// <summary>
        /// Observations to Noise Observations ratio;
        /// </summary>
        private const double Theta = 100;

        /// <summary>
        /// Cluster count;
        /// </summary>
        private readonly int K;

        /// <summary>
        /// Count of observations in each cluster;
        /// </summary>
        private readonly int KN;

        /// <summary>
        /// Dimensionality of the features;
        /// </summary>
        private readonly int M;

        /// <summary>
        /// This parameter is used to ensure a minimum gap between any two clusters.
        /// Range  is (0 , 1] and the value is set randomly.
        /// </summary>
        private readonly double Epsilon;

        // TODO: BitArray?
        private bool[] IsClusterPlaced;
        public float[] ClustersRadius;
        public float[,] ClustersCentroids;

        public SyntheticDataGenerator(
            double maxRadius,
            double minRadius,
            int observationsCount,
            int clustersCount,
            int featuresDim)
        {
            M = featuresDim;
            K = clustersCount;
            N = observationsCount;

            // TODO: Implement noise points generation.
            KN = N / clustersCount;
            P = 0;

            _maxRadius = maxRadius;
            _minRadius = minRadius;

            Epsilon = Random.NextDouble() + 0.0001;
        }

        public IEnumerable<Tuple<int,float[]>> GenerateClusterObservations()
        {
            PlaceClusters();
            var clusterShapes = Enum.GetValues(typeof(ClusterShape));
            for (var i = 0; i < K; i++)
            {
                // Randomly assigning a shape to a cluster.
                var shape = (ClusterShape)clusterShapes.GetValue(Random.Next(clusterShapes.Length));
                using(var pointGetter = GenerateClusterPoints(
                    KN, i, M, ClustersRadius, ClustersCentroids, shape).GetEnumerator())
                {
                    var isNextPoint = pointGetter.MoveNext();

                    while(isNextPoint){
                        yield return new Tuple<int, float[]>(i, pointGetter.Current);
                        isNextPoint = pointGetter.MoveNext();
                    }
                }
            }
        }

        /// <summary>
        /// Places a cluster, so that it space doesn't intersect with the space
        /// of already placed clusters.
        /// </summary>
        private void PlaceCluster(int clusterIndex)
        {
            if (clusterIndex == 0) throw new ArgumentException("First cluster should be placed by default.", nameof(clusterIndex));
            for (var i = 0; i < M; i++)
            {
                var r = ClustersRadius[clusterIndex];
                ClustersCentroids[clusterIndex,i] = (float)Random.NextDouble(r, r + ValueCap);
            }

            do
            {
                IsClusterPlaced[clusterIndex] = true;
                for (var i = 1; i < clusterIndex; i++)
                {
                    var r1 = ClustersRadius[clusterIndex];
                    var r2 = ClustersRadius[i];

                    if(AreClustersSeparated(clusterIndex, i, r1, r2))
                    {
                        IsClusterPlaced[clusterIndex] = false;
                        //TODO: Implement function for reducing the radius by certain rate.
                        ClustersRadius[clusterIndex]--;
                        if (ClustersRadius[clusterIndex] < _minRadius)
                        {
                            PlaceCluster(clusterIndex);
                        }
                        break;
                    }
                }
            }
            while (!IsClusterPlaced[clusterIndex]);
        }

        private bool AreClustersSeparated(int firstClusterIndex, int secondClusterIndex, float r1, float r2)
        {
            var areSeparated = true;

            for(var i = 0; i < M; i++)
            {
                var c1 = ClustersCentroids[firstClusterIndex, i];
                var c2 = ClustersCentroids[secondClusterIndex, i];
                var dist = Math.Abs(c1 - c2);
                if(dist < (1 + Epsilon) * (r1 + r2))
                {
                    areSeparated = false;
                    break;
                }
            }
            return areSeparated;
        }

        /// <summary>
        /// Place all clusters into the feature space.
        /// </summary>
        public void PlaceClusters()
        {
            IsClusterPlaced = new bool[K];
            IsClusterPlaced[0] = true; // First cluster is placed by default.

            ClustersRadius = new float[K];
            for (var i = 0; i < K; i++)
            {
                ClustersRadius[i] = (float)Random.NextDouble(_minRadius, _maxRadius);
            }

            ClustersCentroids = new float[K, M];
            // The first cluster centroid is generated before hand.
            for (var i = 0; i < M; i++)
            {
                var r = ClustersRadius[0];
                ClustersCentroids[0, i] = (float)Random.NextDouble(r, r + ValueCap);
            }

            for (var i = 1; i < K; i++)
            {
                PlaceCluster(i);
            }
        }

        public enum ClusterShape
        {
            Sphere,
            HyperRect,
            Ellipsoid,
            HyperCube,
            // TODO: implement Irregular
        }

        /// <summary>
        /// The cluster points to be generated are representing the final observations
        /// to be provided as an input. Each observation comes from a cluster.
        /// </summary>
        /// <param name="pointsCount"> Count of the points to be generated for the cluster; </param>
        /// <param name="clusterIndex"> Index identifying the cluster where the point is located; </param>
        /// <param name="shape"> Shape of the cluster;</param>
        public static IEnumerable<float[]> GenerateClusterPoints(
            int pointsCount,
            int clusterIndex,
            int featureDim,
            float[] clustersRadius,
            float [,] clustersCentroids,
            ClusterShape shape)
        {
            var r = clustersRadius[clusterIndex];
            var conditionalR = new double[featureDim];
            var gauss = new NormalDistribution(mean: 0, sigma: 1, Random);

            switch (shape)
            {
                case ClusterShape.Sphere:

                    for(var i = 0; i < pointsCount; i++)
                    {
                        var point = SampleWithinSphere(gauss,
                            r,
                            featureDim,
                            clustersCentroids,
                            clusterIndex);

                        yield return point;
                    }
                    break;

                case ClusterShape.Ellipsoid:

                    for (var j = 0; j < conditionalR.Length; j++)
                    {
                        conditionalR[j] = Random.NextDouble(0.1 * r, r);
                    }

                    for (var i = 0; i < pointsCount; i++)
                    {
                        var point = SampleWithinEllipsoid(
                            gauss, 
                            conditionalR,
                            featureDim,
                            clustersCentroids,
                            clusterIndex);

                        yield return point;
                    }
                    break;


                case ClusterShape.HyperCube:
                    for (var i = 0; i < pointsCount; i++)
                    {
                        var point = SampleWithinCube(
                            r,
                            featureDim,
                            clustersCentroids,
                            clusterIndex);

                        yield return point;
                    }
                    break;

                case ClusterShape.HyperRect:

                    for (var j = 0; j < conditionalR.Length; j++)
                    {
                        conditionalR[j] = Random.NextDouble(0.1 * r, r);
                    }

                    for (var i = 0; i < pointsCount; i++)
                    {
                        var point = SampleWithinRect(
                            conditionalR,
                            featureDim,
                            clustersCentroids,
                            clusterIndex);

                        yield return point;
                    }
                    break;
            }
        }

        /// <summary>
        /// Generates the coordinates of a point within a sphere.
        /// https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html
        /// </summary>
        /// <param name="gauss"> Std Normal Dist for generating samples; </param>
        /// <param name="r"> Sphere Radius; </param>
        private static float[] SampleWithinSphere(
            NormalDistribution gauss,
            double r,
            int featureDim,
            float[,] clustersCentroids,
            int clusterIndex)
        {
            var y = gauss.GenerateSample(featureDim);
            var yl2 = y.L2Norm();
            var u = Random.NextDouble();
            var udr = Math.Pow(u, 1.0 / y.Length) * r;

            for (var i = 0; i < y.Length; i++)
            {
                var c = clustersCentroids[clusterIndex, i];
                y[i] = (float)(udr * (y[i] / yl2));
            }

            return y;
        }

        /// <summary>
        /// Generates the coordinates of a point within an ellipsoid.
        /// https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html
        /// </summary>
        /// <param name="gauss"> Std Normal Dist for generating samples; </param>
        /// <param name="conditionalR"> Maximum ellipsoid radius across certain axis; </param>
        private static float[] SampleWithinEllipsoid( 
            NormalDistribution gauss,
            double[] conditionalR,
            int featureDim,
            float[,] clustersCentroids,
            int clusterIndex)
        {
            var y = gauss.GenerateSample(featureDim);
            var yl2 = y.L2Norm();
            var u = Random.NextDouble();
            var udr = new double[conditionalR.Length];

            for (var i = 0; i < udr.Length; i++)
            {
                udr[i] = Math.Pow(u, 1.0 / y.Length) * conditionalR[i];
            }

            for (var i = 0; i < y.Length; i++)
            {
                var c = clustersCentroids[clusterIndex, i];
                y[i] = c + (float)(udr[i] * (y[i] / yl2));
            }

            return y;
        }

        /// <summary> Generates the coordinates of a point in a hypercube; </summary>
        private static float[] SampleWithinCube(double r,
            int featureDim,
            float[,] clustersCentroids,
            int clusterIndex)
        {
            var coordinates = new float[featureDim];
            for(var i = 0; i < coordinates.Length; i++)
            {
                var c = clustersCentroids[clusterIndex, i];
                coordinates[i] = c + (float)Random.NextDouble(0.0001, r);
            }

            return coordinates;
        }

        /// <summary> Generates the coordinates of a point in a hyper-rectangle </summary>
        private static float[] SampleWithinRect(
            double[] conditionalR,
            int featureDim,
            float[,] clustersCentroids,
            int clusterIndex)
        {
            var coordinates = new float[featureDim];

            for (var i = 0; i < coordinates.Length; i++)
            {
                var c = clustersCentroids[clusterIndex, i];
                coordinates[i] = c + (float)Random.NextDouble(0.0001, conditionalR[i]);
            }

            return coordinates;
        }
    }
}
