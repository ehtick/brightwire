﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using BrightData;
using BrightData.UnitTests;
using BrightTable;
using BrightWire.TrainingData.Helper;
using FluentAssertions;
using Xunit;

namespace BrightWire.UnitTests
{
    public class TreeBasedTests : NumericsBase
    {
        [Fact]
        public void TestDecisionTree()
        {
            var stringTableBuilder = new StringTableBuilder();
            var data = NaiveBayesTests.GetSimpleChineseSet(_context, stringTableBuilder)
                .ConvertToWeightedIndexList(_context, false)
                .ConvertToTable(_context)
            ;
            var model = data.TrainDecisionTree();
            var classifier = model.CreateClassifier();
            var testRows = data.AsConvertible().Rows().ToArray();
            classifier.Classify(testRows[0]).GetBestClassification().Should().Be("china");
            classifier.Classify(testRows[1]).GetBestClassification().Should().Be("japan");
        }

        [Fact]
        public void TestRandomForest()
        {
            var stringTableBuilder = new StringTableBuilder();
            var data = NaiveBayesTests.GetSimpleChineseSet(_context, stringTableBuilder)
                .ConvertToWeightedIndexList(_context, false)
                .ConvertToTable(_context);
            var model = data.TrainRandomForest();
            var classifier = model.CreateClassifier();
            var testRows = data.AsConvertible().Rows().ToArray();
            classifier.Classify(testRows[0]).GetBestClassification().Should().Be("china");
        }
    }
}
