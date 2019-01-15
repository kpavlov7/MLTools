using Xunit;
using ML.MathHelpers;
namespace ML.tests
{
    public class ArrayHelperTests
    {
        [Fact]

        public void mean()
        {
            var array = new[] { 1f, 2f, 3f, 4f };

            Assert.Equal(10f / 4f, array.Mean());
        }

        [Fact]
        public void variance_simple()
        {
            var array = new[] { 2f, 1f, 1f, 2f };

            Assert.Equal(0.25, array.Variance());
        }

        [Fact]
        public void variance_zero()
        {
            var array = new[] { 1f, 1f, 1f, 1f };

            Assert.Equal(0, array.Variance());
        }

        [Fact]
        public void sorted_insert_equal()
        {
            var array = new[] { 0, 1.2, };
            var sortedLength = 1;
            array.SortedInsert(sortedLength, 1.2);

            Assert.Equal(array, new[] { 1.2, 1.2 });
        }

        [Fact]
        public void sorted_insert_equal_end()
        {
            var array = new[] { 0, 0, 1.2, };
            var sortedLength = 2;
            array.SortedInsert(sortedLength, 1.2);

            Assert.Equal(array, new[] { 0, 1.2, 1.2 });
        }

        [Fact]
        public void sorted_insert_less()
        {
            var array = new[] { 0, 1.2 };
            var sortedLength = 1;
            array.SortedInsert(sortedLength, 1.19);

            Assert.Equal(array, new[] { 1.19, 1.2 });
        }

        [Fact]
        public void sorted_insert_greater()
        {
            var array = new[] { 0, 1.2 };
            var sortedLength = 1;
            array.SortedInsert(sortedLength, 1.21);

            Assert.Equal(array, new[] { 1.2, 1.21 });
        }

        [Fact]
        public void sorted_insert_medium_less()
        {
            var array = new[] { 0, 0, 0, 1.2, 1.4 };
            var sortedLength = 1;
            array.SortedInsert(sortedLength, 1.3);

            Assert.Equal(array, new[] { 0, 0, 1.2, 1.3, 1.4 });
        }

        [Fact]
        public void sorted_insert_medium_greater()
        {
            var array = new[] { 0, 0, 0, 1.2, 1.4 };
            var sortedLength = 1;
            array.SortedInsert(sortedLength, 1.5);

            Assert.Equal(array, new[] { 0, 0, 1.2, 1.4, 1.5 });
        }

        [Fact]
        public void sorted_insert_medium_equal_beginning()
        {
            var array = new[] { 0, 0, 0, 1.2, 1.4 };
            var sortedLength = 1;
            array.SortedInsert(sortedLength, 1.2);

            Assert.Equal(array, new[] { 0, 0, 1.2, 1.2, 1.4 });
        }

        [Fact]
        public void sorted_insert_medium_equal_end()
        {
            var array = new[] { 0, 0, 0, 1.2, 1.4 };
            var sortedLength = 1;
            array.SortedInsert(sortedLength, 1.4);

            Assert.Equal(array, new[] { 0, 0, 1.2, 1.4, 1.4 });
        }
    }
}
