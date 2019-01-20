using System;
using ML.MathHelpers;

namespace ML.Clustering
{
    public class SelfOrganizingMap
    {

        public ushort[] BMUCoordinates { get; set; }
        private int FeaturesCount { get; set; }

        public int _iterationsCount;
        private ushort[] _gridDimensions;
        private double _initialNeighbourRadius;
        private float[,] _weights;
        private double _initialLearningRate;

        /// <summary>
        /// Indicator whether we should use k-means++ initialization.
        /// If false we use random initialization.
        /// </summary>
        private bool _usePlusPlusInit;

        private readonly Func<int, double> LearningRate;
        private readonly Func<int, double> NeighbourhoodRadius;

        /// <summary>
        /// Random generator seed;
        /// </summary>
        private const int Seed = 12;
        private Random _random;

        public SelfOrganizingMap(
            double initialLearningRate,
            ushort[] gridDimensions,
            int iterationsCount,
            float initialNeighbourRadius,
            bool usePlusPlusInit = false)
        {
            _initialLearningRate = initialLearningRate;
            _iterationsCount = iterationsCount;
            _gridDimensions = gridDimensions;
            _iterationsCount = iterationsCount;
            _initialNeighbourRadius = initialNeighbourRadius;
            _usePlusPlusInit = usePlusPlusInit;

            // TODO: provide them as input;
            LearningRate = t => _initialLearningRate * Math.Exp(t / (double)_iterationsCount);
            NeighbourhoodRadius = t => initialNeighbourRadius * Math.Exp(- t / (double)_iterationsCount);

            _random = new Random(Seed);
        }

        public int[] Train(InstanceRepresentation set)
        {


            set.Standardize();
            var instances = set.Instances.ToArray();

            _weights = new float[_gridDimensions[0] * _gridDimensions[1], set.FeauturesCount];
            FeaturesCount = set.FeauturesCount;

            // Intialize weights with gaussians
            var n = _gridDimensions[0] * _gridDimensions[1];

            if (!_usePlusPlusInit || n >= instances.Length)
            {
                for (ushort i = 0; i < n; i++)
                {
                    for (ushort j = 0; j < FeaturesCount; j++)
                    {
                        _weights[i, j] = (float)GaussHelper.InvPhi(_random.NextDouble());
                    }
                }
            }
            else
            {
                _weights = PlusPlusInitializer.InitializeCentroids(n, instances, _random);
            }

            for (var i = 0; i < _iterationsCount; i++)
            {
                var instance = instances[_random.Next(instances.Length)];
                // Best Matching Unit
                var bmuIndex = Instances.MinEucDistanceIndex(instance, _weights);
                BMUCoordinates = ToCoordinates(bmuIndex);
                UpdateHexagonWeights((ushort)NeighbourhoodRadius(i),LearningRate(i), BMUCoordinates, instance.GetValues());
            }

            var instancesClusters = new int[instances.Length];
            for (var i = 0; i < instances.Length; i++)
            {
                instancesClusters[i] = Instances.MinEucDistanceIndex(instances[i], _weights);
            }

            return instancesClusters;
        }

        public ushort[] ToCoordinates(int index)
        {
            var y = (ushort)(index % _gridDimensions[1]);
            var x = (ushort)(index / _gridDimensions[1]);

            return new ushort[] {x, y};
        }

        public void UpdateHexagonWeights(int r, double lambda, ushort[] coordinates, float[] instance)
        {
            var n = 6;

            for(ushort direction = 0; direction < n; direction++)
            {
                UpdateHexDirection(r, lambda, direction, coordinates, instance);
            }
        }
        public void UpdateHexDirection(int r, double lambda, ushort direction, ushort[] coordinates, float[] instance)
        {
            if(
                coordinates[0] >= 0 && coordinates[1] >= 0 &&
                coordinates[0] < _gridDimensions[0] && coordinates[1] < _gridDimensions[1]
                )
            {
                var bmuIndex = coordinates[0] * (_gridDimensions[1] - 1) + coordinates[1];
                var h = NeighbourhoodFunction(r, lambda, coordinates);

                for (var i = 0; i < FeaturesCount; i++)
                {
                    _weights[bmuIndex, i] += (float)(lambda * h * (instance[i] - _weights[bmuIndex, i]));
                }
            }

            r--;

            if (r >= 0)
            {
                var nextCoordinates = HexagonalNeighbourhood.GetNeighbour(direction, coordinates);
                UpdateHexDirection(r, lambda, direction, nextCoordinates, instance);
            }
        }
        public double NeighbourhoodFunction(int r, double lambda, ushort[] coordinates)
        {
            var coordL2Norm = 0.0;

            for(var i = 0; i < coordinates.Length; i++)
            {
                coordL2Norm += (BMUCoordinates[i] - coordinates[i]) * (BMUCoordinates[i] - coordinates[i]);
            }
            coordL2Norm = Math.Sqrt(coordL2Norm);
            return lambda * Math.Exp(-coordL2Norm / (2.0 * (r * r)));
        }
    }
}
