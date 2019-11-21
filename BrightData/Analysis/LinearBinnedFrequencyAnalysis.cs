﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace BrightData.Analysis
{
    public class LinearBinnedFrequencyAnalysis
    {
        readonly double _min, _max, _step;
        readonly ulong[] _bins;
        ulong _belowRange = 0, _aboveRange = 0;

        public LinearBinnedFrequencyAnalysis(double min, double max, uint numBins)
        {
            _step = (max - min) / numBins;
            _min = min;
            _max = max;

            _bins = new ulong[numBins];
        }

        public void Add(double val)
        {
            if (val < _min)
                _belowRange++;
            else if (val > _max)
                _aboveRange++;
            else {
                var binIndex = Convert.ToInt32((val - _min) / _step);
                if (binIndex >= _bins.Length)
                    --binIndex;
                _bins[binIndex]++;
            }
        }

        public IReadOnlyList<(double Start, double End, ulong Count)> ContinuousFrequency
        {
            get 
            {
                var ret = new List<(double Start, double End, ulong Count)>();
                if (_belowRange > 0)
                    ret.Add((double.NegativeInfinity, _min, _belowRange));
                ret.AddRange(_bins.Select((c, i) => (
                    _min + (i * _step), 
                    _min + (i + 1) * _step, 
                    c
                )));
                if(_aboveRange > 0)
                    ret.Add((_max, double.PositiveInfinity, _aboveRange));
                return ret;
            }
        }
    }
}
