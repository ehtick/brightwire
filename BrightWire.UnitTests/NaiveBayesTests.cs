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
    public class NaiveBayesTests : NumericsBase
    {
        [Fact]
        public void TestNaiveBayes()
        {
            var dataTable = _context.BuildTable();
            dataTable.AddColumn(ColumnType.Float, "height");
            dataTable.AddColumn(ColumnType.Int, "weight");
            dataTable.AddColumn(ColumnType.Int, "foot-size");
            dataTable.AddColumn(ColumnType.String, "gender").SetTargetColumn(true);

            // sample data from: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
            dataTable.AddRow(6f, 180, 12, "male");
            dataTable.AddRow(5.92f, 190, 11, "male");
            dataTable.AddRow(5.58f, 170, 12, "male");
            dataTable.AddRow(5.92f, 165, 10, "male");
            dataTable.AddRow(5f, 100, 6, "female");
            dataTable.AddRow(5.5f, 150, 8, "female");
            dataTable.AddRow(5.42f, 130, 7, "female");
            dataTable.AddRow(5.75f, 150, 9, "female");
            var index = dataTable.Build();

            var testData = _context.BuildTable();
            testData.CopyColumnsFrom(index);
            testData.AddRow(6f, 130, 8, "?");
            var testDataTable = testData.Build().AsConvertible();

            var model = index.TrainNaiveBayes();
            var classifier = model.CreateClassifier();
            var classification = classifier.Classify(testDataTable.Row(0));
            classification.First().Label.Should().Be("female");
        }

        public static IReadOnlyList<(string Label, IndexList Data)> GetSimpleChineseSet(IBrightDataContext context, StringTableBuilder stringTableBuilder)
        {
            // sample data from: http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
            var data = new[] {
                (new[] { "Chinese", "Beijing", "Chinese" }, true),
                (new[] { "Chinese", "Chinese", "Shanghai" }, true),
                (new[] { "Chinese", "Macao" }, true),
                (new[] { "Tokyo", "Japan", "Chinese" }, false),
            };

            return data.Select(r => (r.Item2 ? "china" : "japan", IndexList.Create(
                context,
                r.Item1.Select(stringTableBuilder.GetIndex).ToArray()
            ))).ToList();
        }

        public static IndexList GetTestRow(IBrightDataContext context, StringTableBuilder stringTableBuilder)
        {
            return IndexList.Create(context,
                new[] {"Chinese", "Chinese", "Chinese", "Tokyo", "Japan"}.Select(s => stringTableBuilder.GetIndex(s))
                    .ToArray()
            );
        }

        [Fact]
        public void TestMultinomialNaiveBayes()
        {
            var stringTableBuilder = new StringTableBuilder();
            var data = GetSimpleChineseSet(_context, stringTableBuilder);
            var model = data.TrainMultinomialNaiveBayes();
            var classifier = model.CreateClassifier();
            var classification = classifier.Classify(GetTestRow(_context, stringTableBuilder));
            classification.OrderByDescending(c => c.Weight).First().Label.Should().Be("china");
        }

        [Fact]
        public void TestBernoulliNaiveBayes()
        {
            var stringTableBuilder = new StringTableBuilder();
            var data = GetSimpleChineseSet(_context, stringTableBuilder);
            var model = data.TrainBernoulliNaiveBayes();
            var classifier = model.CreateClassifier();
            var classification = classifier.Classify(GetTestRow(_context, stringTableBuilder));
            classification.OrderByDescending(c => c.Weight).First().Label.Should().Be("japan");
        }
    }
}