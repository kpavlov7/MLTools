using System;
using System.Collections.Generic;
using System.Text;

namespace ML
{
    public static class HexagonalNeighbourhood
    {
        public static short[,,] HexagonalDirections = new short[,,] {
            { { 1, 0 }, { 0, -1}, { -1, -1}, { -1, 0 }, { -1, 1}, { 0, 1} },
            { { 1, 0 }, { 1, -1}, { 0, -1}, { -1, 0 }, { 0, 1}, { 1, 1} }
        };

        public static ushort[] GetNeighbour(int direction, ushort[] coordinates)
        {
            var parity = coordinates[0] % 1; //Odd or even row
            var dir = new short[2];
            dir[0] = HexagonalDirections[parity, direction, 0];
            dir[1] = HexagonalDirections[parity, direction, 0];
            var neighbour = new ushort[2];

            //TODO: checking whether dir is negative!
            neighbour[0] = (ushort)(coordinates[0] + dir[0]);
            neighbour[1] = (ushort)(coordinates[1] + dir[1]);
            return neighbour;
        }
    }
}