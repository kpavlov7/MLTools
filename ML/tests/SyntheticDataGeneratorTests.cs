using System.IO;
using Xunit;

namespace ML.tests
{
    public class SyntheticDataGeneratorTests
    {
        private const string path = "";
        private const string csvFileName = "cluster_points.csv";

        //TODO: Replace the visual check with something efficient for unit test;
        [Fact]
        public void place_clusters() {
            var maxRadius = 100;
            var minRadius = 10;
            var clusterCount = 10;
            var featureDim = 20;
            var obsCount = 1000;

            var clusterGenerator = new SyntheticDataGenerator(maxRadius, minRadius, obsCount, clusterCount, featureDim);
            //using (var textWriter = new StreamWriter(path + csvFileName))
            using (var obsGetter = clusterGenerator.GenerateClusterObservations().GetEnumerator())
            {
                var isNextObservation = obsGetter.MoveNext();

                while (isNextObservation)
                {
                    var obs = obsGetter.Current.Item2;
                    //textWriter.WriteLine(string.Join(',', obs));
                    isNextObservation = obsGetter.MoveNext();
                }
            }
        }
    }
}
