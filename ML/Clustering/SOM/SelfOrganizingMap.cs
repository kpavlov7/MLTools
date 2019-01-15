using System;
using ML.MathHelpers;

namespace ML
{
    public class SelfOrganizingMap
    {
        public double InitialLearningRate { get; set; }
        public ushort[] GridDimensions { get; set; }
        public ushort[] BMUCoordinates { get; set; }
        public int IterationsCount { get; set; }
        public double InitialNeighbourRadius { get; set; }
        public float [,] Weights { get; set; }

        public Func<int, double> LearningRate;
        public Func<int, double> NeighbourhoodRadius;

        private int _featuresCount { get; set; }

        /// <summary>
        /// Random generator seed;
        /// </summary>
        private const int Seed = 12;

        public SelfOrganizingMap(
            double initialLearningRate,
            ushort[] gridDimensions,
            int iterationsCount,
            float initialNeighbourRadius)
        {
            InitialLearningRate = initialLearningRate;
            IterationsCount = iterationsCount;
            GridDimensions = gridDimensions;
            IterationsCount = iterationsCount;
            InitialNeighbourRadius = initialNeighbourRadius;

            // TODO: provide them as input;
            LearningRate = t => InitialLearningRate * Math.Exp(t / (double)IterationsCount);
            NeighbourhoodRadius = t => initialNeighbourRadius * Math.Exp(- t / (double)IterationsCount);
        }

        public int[] Train(InstanceRepresentation set)
        {
            var random = new Random(Seed);

            set.Standardize();

            Weights = new float[GridDimensions[0] * GridDimensions[1], set.FeauturesCount];
            _featuresCount = set.FeauturesCount;

            // Intialize weights with gaussians
            var n = GridDimensions[0] * GridDimensions[1];

            var k = 0;
            for (ushort i = 0; i < n; i++)
            {
                for (ushort j = 0; j < _featuresCount; j++)
                {
                    Weights[i, j] = (float)GaussHelper.InvPhi(random.NextDouble());
                    k++;
                }
            }

            for (var i = 0; i < IterationsCount; i++)
            {
                var instance = set.DrawRandomInstance(random);
                // Best Matching Unit
                var bmuIndex = set.MinEucDistanceIndex(instance, Weights);
                BMUCoordinates = ToCoordinates(bmuIndex);
                UpdateHexagonWeights((ushort)NeighbourhoodRadius(i),LearningRate(i), BMUCoordinates, instance.GetValues());
            }

            var instancesClusters = new int[set.Instances.Count];
            for (var i = 0; i < set.Instances.Count; i++)
            {
                instancesClusters[i] = set.MinEucDistanceIndex(set.Instances[i], Weights);
            }

            return instancesClusters;
        }

        public ushort[] ToCoordinates(int index)
        {
            var y = (ushort)(index % GridDimensions[1]);
            var x = (ushort)(index / GridDimensions[1]);

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
                coordinates[0] < GridDimensions[0] && coordinates[1] < GridDimensions[1]
                )
            {
                var bmuIndex = coordinates[0] * (GridDimensions[1] - 1) + coordinates[1];
                var h = NeighbourhoodFunction(r, lambda, coordinates);

                for (var i = 0; i < _featuresCount; i++)
                {
                    Weights[bmuIndex, i] += (float)(lambda * h * (instance[i] - Weights[bmuIndex, i]));
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
