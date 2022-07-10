﻿using BrightData.LinearAlgebra;
using MKLNET;

namespace BrightData.MKL
{
    public class MklVector : Vector<MklLinearAlgebraProvider>
    {
        public MklVector(ITensorSegment2 data, MklLinearAlgebraProvider computationUnit) : base(data, computationUnit)
        {
        }

        public override IVector Create(ITensorSegment2 segment) => new MklVector(segment, _lap);
    }
}