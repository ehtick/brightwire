﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using BrightData;
using BrightData.Distributions;
using BrightTable;
using BrightWire;
using BrightWire.Models.Bayesian;
using BrightWire.TrainingData.Helper;

namespace ExampleCode.DataTableTrainers
{
    class SentenceTable
    {
        readonly IRowOrientedDataTable _sentenceTable;
        readonly Dictionary<string, uint> _stringIndex = new Dictionary<string, uint>();
        readonly List<string> _strings = new List<string>();

        public uint GetStringIndex(string str)
        {
            if (!_stringIndex.TryGetValue(str, out var ret)) {
                _stringIndex.Add(str, ret = (uint)_strings.Count);
                _strings.Add(str);
            }

            return ret;
        }

        public string GetString(uint stringIndex) => _strings[(int) stringIndex];

        public SentenceTable(IBrightDataContext context, IEnumerable<string[]> sentences)
        {
            var builder = context.BuildTable();
            builder.AddColumn(ColumnType.IndexList, "Sentences");
            foreach(var sentence in sentences)
                builder.AddRow(context.CreateIndexList(sentence.Select(GetStringIndex).ToArray()));
            _sentenceTable = builder.Build();
        }

        (uint Index, string String) Append(uint index, StringBuilder sb)
        {
            var str = GetString(index);
            if (Char.IsLetterOrDigit(str[0]) && sb.Length > 0) {
                var lastChar = sb[^1];
                if (lastChar != '\'' && lastChar != '-')
                    sb.Append(' ');
            }
            sb.Append(str);
            return (index, str);
        }

        public MarkovModel3<uint> TrainMarkovModel(bool writeResults = true)
        {
            // create a markov trainer that uses a window of size 3
            var context = _sentenceTable.Context;
            var trainer = context.CreateMarkovTrainer3<uint>();
            foreach(var sentence in _sentenceTable.Column<IndexList>(0).EnumerateTyped())
                trainer.Add(sentence.Indices);

            var ret = trainer.Build();
            if (writeResults) {
                foreach(var sentence in GenerateText(ret))
                    Console.WriteLine(sentence);
            }

            return ret;
        }

        public IEnumerable<string> GenerateText(MarkovModel3<uint> model, int count = 50)
        {
            var context = _sentenceTable.Context;
            var table = model.AsDictionary;

            for (var i = 0; i < 50; i++) {
                var sb = new StringBuilder();
                uint prevPrev = default, prev = default, curr = default;
                for (var j = 0; j < 1024; j++) {
                    var transitions = table.GetTransitions(prevPrev, prev, curr);
                    var distribution = new CategoricalDistribution(context, transitions.Select(d => d.Probability));
                    var next = Append(transitions[distribution.Sample()].NextState, sb);

                    if (SimpleTokeniser.IsEndOfSentence(next.String))
                        break;
                    prevPrev = prev;
                    prev = curr;
                    curr = next.Index;
                }
                yield return sb.ToString();
            }
        }
    }
}