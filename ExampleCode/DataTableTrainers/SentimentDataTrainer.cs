﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrightData;
using BrightData.Helper;
using BrightData.LinearAlgebra;
using BrightWire;
using BrightWire.ExecutionGraph;
using BrightWire.Models;
using BrightWire.Models.Bayesian;
using BrightWire.TrainingData.Helper;

namespace ExampleCode.DataTableTrainers
{
    internal class SentimentDataTrainer
    {
        readonly (string Classification, IndexList Data)[] _indexedSentencesTraining;
        readonly (string Classification, IndexList Data)[] _indexedSentencesTest;
        readonly IBrightDataContext _context;
        readonly StringTableBuilder _stringTable;
        readonly uint _maxIndex;

        public SentimentDataTrainer(IBrightDataContext context, DirectoryInfo directory)
        {
            var files = new[]
            {
                "amazon_cells_labelled.txt",
                "imdb_labelled.txt",
                "yelp_labelled.txt"
            };
            var lineSeparator = "\n".ToCharArray();
            var separator = "\t".ToCharArray();
            _stringTable = new StringTableBuilder();

            var sentences = new List<(string[] Sentence, string Classification)>();
            foreach (var path in files.Select(f => Path.Combine(directory.FullName, "sentiment labelled sentences", f)))
            {
                var data = File.ReadAllText(path)
                    .Split(lineSeparator)
                    .Where(l => !String.IsNullOrWhiteSpace(l))
                    .Select(l => l.Split(separator))
                    .Select(s => (Sentence: Tokenise(s[0]), Classification: s[1][0] == '1' ? "positive" : "negative"))
                    .Where(d => d.Sentence.Any());
                sentences.AddRange(data);
            }

            var (training, test) = sentences.Shuffle(context.Random).ToArray().Split();
            _indexedSentencesTraining = BuildIndexedClassifications(context, training, _stringTable);
            _indexedSentencesTest = BuildIndexedClassifications(context, test, _stringTable);
            _maxIndex = _indexedSentencesTraining.Concat(_indexedSentencesTest).Max(d => d!.Data.Indices.Max());
            _context = context;
        }

        public StringTable StringTable => _stringTable.StringTable;

        public BernoulliNaiveBayes TrainBernoulli()
        {
            var bernoulli = _indexedSentencesTraining.TrainBernoulliNaiveBayes();
            Console.WriteLine("Bernoulli accuracy: {0:P}", _indexedSentencesTest
                .Classify(bernoulli.CreateClassifier())
                .Average(r => r.Score)
            );
            return bernoulli;
        }

        public MultinomialNaiveBayes TrainMultinomialNaiveBayes()
        {
            var multinomial = _indexedSentencesTraining.TrainMultinomialNaiveBayes();
            Console.WriteLine("Multinomial accuracy: {0:P}", _indexedSentencesTest
                .Classify(multinomial.CreateClassifier())
                .Average(r => r.Score)
            );
            return multinomial;
        }

        public (IGraphExecutionEngine, WireBuilder, IGraphExecutionEngine) TrainNeuralNetwork(uint numIterations = 10)
        {
            var indexer = GetIndexer();
            var trainingTable = GetTable(_context, _maxIndex, indexer, _indexedSentencesTraining);
            var testTable = GetTable(_context, _maxIndex, indexer, _indexedSentencesTest);
            var graph = _context.CreateGraphFactory();

            var trainingData = graph.CreateDataSource(trainingTable);
            var testData = graph.CreateDataSource(testTable);

            // use rmsprop gradient descent and xavier weight initialisation
            var errorMetric = graph.ErrorMetric.BinaryClassification;
            graph.CurrentPropertySet
                .Use(graph.GradientDescent.RmsProp)
                .Use(graph.WeightInitialisation.Xavier)
            ;

            var engine = graph.CreateTrainingEngine(trainingData, errorMetric, 0.3f);
            engine.LearningContext.ScheduleLearningRate(10, 0.2f);

            var neuralNetworkWire = graph.Connect(engine)
                .AddFeedForward(512, "layer1")
                //.AddBatchNormalisation()
                .Add(graph.ReluActivation())
                //.AddDropOut(0.5f)
                .AddFeedForward(trainingData.GetOutputSizeOrThrow(), "layer2")
                .Add(graph.ReluActivation())
                .AddBackpropagation("first-network")
            ;

            Console.WriteLine("Training neural network classifier...");
            GraphModel? bestNetwork = null;
            engine.Train(numIterations, testData, network => bestNetwork = network);
            var firstClassifier = graph.CreateExecutionEngine(engine.Graph);
            return (engine.CreateExecutionEngine(bestNetwork?.Graph), neuralNetworkWire, firstClassifier);
        }

        public IGraphEngine StackClassifiers(IGraphTrainingEngine engine, WireBuilder neuralNetworkWire, IIndexListClassifier bernoulli, IIndexListClassifier multinomial)
        {
            // create combined data tables with both index lists and encoded vectors
            var graph = engine.LearningContext.GraphFactory;
            var context = graph.Context;
            var errorMetric = graph.ErrorMetric.BinaryClassification;
            var maxIndex = _indexedSentencesTraining.Concat(_indexedSentencesTest).Max(d => d!.Data.Indices.Max());
            var indexer = GetIndexer();
            var training = CreateCombinedDataTable(context, maxIndex, indexer, _indexedSentencesTraining);
            var test = CreateCombinedDataTable(context, maxIndex, indexer, _indexedSentencesTest);
            var trainingData = graph.CreateDataSource(training, 0);
            var testData = trainingData.CloneWith(test);
            var outputSize = trainingData.GetOutputSizeOrThrow();

            // stop the backpropagation to the first neural network
            engine.LearningContext.EnableNodeUpdates(neuralNetworkWire.Find("layer1")!, false);
            engine.LearningContext.EnableNodeUpdates(neuralNetworkWire.Find("layer2")!, false);

            // create the bernoulli classifier wire
            var bernoullWireToNode = graph.Connect(engine)
                .AddClassifier(bernoulli.AsRowClassifier(1, indexer), training)
            ;

            // create the multinomial classifier wire
            var multinomialWire = graph.Connect(engine)
                .AddClassifier(multinomial.AsRowClassifier(1, indexer), training)
            ;

            // join the bernoulli, multinomial and neural network classification outputs
            var firstNetwork = neuralNetworkWire.Find("first-network")!;
            var joined = graph.Join(multinomialWire, graph.Join(bernoullWireToNode, graph.Connect(outputSize, firstNetwork)));

            // train an additional classifier on the output of the previous three classifiers
            joined
                .AddFeedForward(outputSize: 64)
                .Add(graph.ReluActivation())
                .AddDropOut(dropOutPercentage: 0.5f)
                .AddFeedForward(outputSize)
                .Add(graph.ReluActivation())
                .AddBackpropagation()
            ;

            // train the network again
            Console.WriteLine("Training stacked neural network classifier...");
            GraphModel? bestStackedNetwork = null;
            engine.Train(20, testData, network => bestStackedNetwork = network);
            if (bestStackedNetwork != null)
                engine.LoadParametersFrom(graph, bestStackedNetwork.Graph);

            return graph.CreateExecutionEngine(engine.Graph);
        }

        static IRowOrientedDataTable GetTable(IBrightDataContext context, uint maxIndex, IIndexStrings indexer, (string Classification, IndexList Data)[] data)
        {
            var builder = context.BuildTable();
            builder.AddColumn(ColumnType.Vector, "Features");
            builder.AddColumn(ColumnType.Vector, "Target").SetTarget(true);

            var vector = new float[1];
            foreach (var (classification, indexList) in data) {
                var features = indexList.ToDense(maxIndex);
                vector[0] = Convert.ToSingle(indexer.GetIndex(classification));
                builder.AddRow(features, context.CreateVector(vector));
            }

            return builder.BuildRowOriented();
        }

        static string[] Tokenise(string str) => SimpleTokeniser.JoinNegations(SimpleTokeniser.Tokenise(str).Select(s => s.ToLower())).ToArray();

        static (string Classification, IndexList Data)[] BuildIndexedClassifications(IBrightDataContext context, (string[], string)[] data, StringTableBuilder stringTable)
        {
            return data
                .Select(d => (d.Item2, context.CreateIndexList(d.Item1.Select(stringTable.GetIndex).ToArray())))
                .ToArray()
            ;
        }

        static IRowOrientedDataTable CreateCombinedDataTable(IBrightDataContext context, uint maxIndex, IIndexStrings indexer, (string Classification, IndexList Data)[] data)
        {
            var builder = context.BuildTable();
            builder.AddColumn(ColumnType.Vector, "Vector");
            builder.AddColumn(ColumnType.IndexList, "Index List");
            builder.AddColumn(ColumnType.String, "Target");
            builder.AddColumn(ColumnType.Vector, "Vector Target").SetTarget(true);

            var vector = new float[1];
            foreach (var (classification, indexList) in data) {
                var features = indexList.ToDense(maxIndex);
                vector[0] = Convert.ToSingle(indexer.GetIndex(classification));
                builder.AddRow(features, indexList, classification, context.CreateVector(vector));
            }

            return builder.BuildRowOriented();
        }

        static IIndexStrings GetIndexer() => StringIndexer.Create("negative", "positive");

        public void TestClassifiers(IIndexListClassifier bernoulli, IIndexListClassifier multinomial, IGraphExecutionEngine neuralNetwork)
        {
            Console.WriteLine("Enter some text to test the classifiers...");
            while (true)
            {
                Console.Write(">");
                var line = Console.ReadLine();
                if (String.IsNullOrWhiteSpace(line))
                    break;

                var tokens = Tokenise(line);
                var indices = new List<uint>();
                foreach (var token in tokens)
                {
                    if (_stringTable.TryGetIndex(token, out uint stringIndex))
                        indices.Add(stringIndex);
                }
                if (indices.Any())
                {
                    var queryTokens = indices.GroupBy(d => d).Select(g => Tuple.Create(g.Key, (float)g.Count())).ToList();
                    var vector = new float[_maxIndex+1];
                    foreach (var token in queryTokens)
                        vector[token.Item1] = token.Item2;
                    var indexList2 = _context.CreateIndexList(indices);
                    var encodedInput = indexList2.ToDense(_maxIndex).ToArray();

                    Console.WriteLine("Bernoulli classification: " + bernoulli.Classify(indexList2).First().Label);
                    Console.WriteLine("Multinomial classification: " + multinomial.Classify(indexList2).First().Label);
                    var result = neuralNetwork.Execute(encodedInput);
                    if (result != null) {
                        var output = result.Output[0][0];
                        var label = output >= 0.5f ? "positive" : "negative";
                        Console.WriteLine($"Neural network classification: {label} ({output})");
                    }
                }
                else
                    Console.WriteLine("Sorry, none of those words have been seen before.");
                Console.WriteLine();
            }
        }

        public void TrainBiRecurrent()
        {
            var graph = _context.CreateGraphFactory();
            var trainingTable = CreateTable(_indexedSentencesTraining);
            var testTable = CreateTable(_indexedSentencesTest);
            var training = graph.CreateDataSource(trainingTable);
            var test = training.CloneWith(testTable);
            var errorMetric = graph.ErrorMetric.BinaryClassification;
            var engine = graph.CreateTrainingEngine(training, errorMetric, learningRate: 0.001f, batchSize: 128);

            graph.CurrentPropertySet
                .Use(graph.RmsProp())
                .Use(graph.WeightInitialisation.Gaussian)
            ;

            // build the network
            const int HIDDEN_LAYER_SIZE = 128;

            var forward = graph.Connect(engine)
                .AddSimpleRecurrent(graph.TanhActivation(), new float[HIDDEN_LAYER_SIZE], "forward")
            ;
            var reverse = graph.Connect(engine)
                .ReverseSequence()
                .AddSimpleRecurrent(graph.TanhActivation(), new float[HIDDEN_LAYER_SIZE], "backward")
            ;
            graph.BidirectionalJoin(forward, reverse)
                .AddFeedForward(engine.DataSource.GetOutputSizeOrThrow(), "joined")
                .Add(graph.SigmoidActivation())
                .AddBackpropagationThroughTime()
            ;

            ExecutionGraphModel? bestGraph = null;
            engine.Train(20, test, bn => bestGraph = bn.Graph);
        }

        IRowOrientedDataTable CreateTable((string Classification, IndexList Data)[] data)
        {
            var builder = _context.BuildTable();
            builder.AddColumn(ColumnType.Matrix);
            builder.AddColumn(ColumnType.Matrix).SetTarget(true);

            var empty = new float[100];
            foreach (var row in data) {
                var input = row.Data.Indices.Select(i => _context.CreateVector(Data.Embeddings.Get(_stringTable.GetString(i)) ?? empty)).ToArray();
                var output = _context.CreateMatrix((uint)input.Length, 1, row.Classification == "positive" ? 1f : 0f);
                
                builder.AddRow(_context.CreateMatrixFromRows(input), output);
            }

            return builder.BuildRowOriented();
        }
    }
}
