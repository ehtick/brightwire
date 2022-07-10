﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;

namespace BrightData.Numerics
{
    internal class NumericsMatrix2 : LinearAlgebra.Matrix<NumericsLinearAlgebraProvider>
    {
        readonly MathNet.Numerics.LinearAlgebra.Matrix<float> _matrix;

        public NumericsMatrix2(ITensorSegment2 data, uint rows, uint columns, NumericsLinearAlgebraProvider computationUnit) : base(data, rows, columns, computationUnit)
        {
            _matrix = DenseMatrix.Build.Dense((int)RowCount, (int)ColumnCount);
        }

        public override IMatrix Create(ITensorSegment2 segment) => new NumericsMatrix2(segment, RowCount, ColumnCount, _lap);
    }
}
