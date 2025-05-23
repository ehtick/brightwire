﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BrightData;
using BrightData.Types;
using BrightWire;
using BrightWire.Models.Bayesian;
using BrightWire.TrainingData.Helper;

namespace ExampleCode.DataTableTrainers
{
    internal class SentenceTable
    {
        readonly IDataTable _sentenceTable;
        readonly Dictionary<string, uint> _stringIndex = [];
        readonly List<string> _strings = [];
        readonly uint _empty;

        public SentenceTable(BrightDataContext context, IEnumerable<string[]> sentences)
        {
            // create an empty string to represent null
            _empty = GetStringIndex("");

            var builder = context.CreateTableBuilder();
            builder.CreateColumn(BrightDataType.IndexList, "Sentences");
            foreach(var sentence in sentences)
                builder.AddRow(IndexList.Create(sentence.Select(GetStringIndex).ToArray()));
            _sentenceTable = builder.BuildInMemory().Result;
        }

        public uint GetStringIndex(string str)
        {
            if (!_stringIndex.TryGetValue(str, out var ret)) {
                _stringIndex.Add(str, ret = (uint)_strings.Count);
                _strings.Add(str);
            }

            return ret;
        }

        (uint Index, string String) Append(uint index, StringBuilder sb)
        {
            var str = _strings[(int)index];
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
            var trainer = context.CreateMarkovTrainer3(_empty);
            var column = _sentenceTable.GetColumn<IndexList>(0);
            foreach(var sentence in column.EnumerateAllTyped().ToBlockingEnumerable())
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

            for (var i = 0; i < count; i++) {
                var sb = new StringBuilder();
                uint prevPrev = 0, prev = 0, curr = 0;
                for (var j = 0; j < 1024; j++) {
                    var transitions = table.GetTransitions(prevPrev, prev, curr);
                    if (transitions == null)
                        break;
                    var distribution = context.CreateCategoricalDistribution(transitions.Select(d => d.Probability));
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
